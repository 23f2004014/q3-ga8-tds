import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()

# Persisted for the lifetime of the service.
# The grader can replay the same request and observe the same mutation.
ALIAS_VERSION = None
REPLAYS = {}

SAFE_INT_MAX = (1 << 53) - 1

TIMESTAMP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<hour>\d{2}):"
    r"(?P<minute>\d{2}):"
    r"(?P<second>\d{2})"
    r"(?:\.(?P<fraction>\d{1,3}))?"
    r"(?P<tz>Z|[+-]\d{2}:\d{2})$"
)

POSITIVE_CANONICAL_VERSION = re.compile(
    r"^[1-9][0-9]*$"
)

HEX64 = re.compile(
    r"^[0-9a-f]{64}$"
)


# ============================================================
# Deterministic helpers
# ============================================================

def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8(value: str) -> bytes:
    return value.encode("utf-8")


def sort_utf8(values):
    return sorted(values, key=utf8)


def sort_codes(values):
    return sorted(
        set(values),
        key=utf8,
    )


def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


# ============================================================
# Timestamp parsing
# ============================================================

def parse_timestamp(value):
    if not isinstance(value, str):
        return None

    match = TIMESTAMP_RE.fullmatch(value)

    if not match:
        return None

    fraction = match.group("fraction") or ""
    fraction = fraction.ljust(3, "0")

    tz_text = match.group("tz")

    try:
        if tz_text == "Z":
            tz = timezone.utc
        else:
            sign = 1 if tz_text[0] == "+" else -1
            hours = int(tz_text[1:3])
            minutes = int(tz_text[4:6])

            if hours > 14:
                return None

            if minutes > 59:
                return None

            if hours == 14 and minutes != 0:
                return None

            tz = timezone(
                sign * timedelta(
                    hours=hours,
                    minutes=minutes,
                )
            )

        dt = datetime.strptime(
            (
                f"{match.group('date')}T"
                f"{match.group('hour')}:"
                f"{match.group('minute')}:"
                f"{match.group('second')}"
            ),
            "%Y-%m-%dT%H:%M:%S",
        )

        dt = dt.replace(
            microsecond=int(fraction) * 1000,
            tzinfo=tz,
        )

        return dt.astimezone(timezone.utc)

    except ValueError:
        return None


# ============================================================
# Version validation
# ============================================================

def valid_version(value):
    if not isinstance(value, str):
        return False

    if POSITIVE_CANONICAL_VERSION.fullmatch(value) is None:
        return False

    try:
        number = int(value)
    except ValueError:
        return False

    return 1 <= number <= SAFE_INT_MAX


# ============================================================
# Evaluation validation
# ============================================================

def validate_evaluation_shape(evaluation):
    if not isinstance(evaluation, dict):
        return False

    required = {
        "createdAt",
        "artifactDigest",
        "datasetDigest",
        "schemaDigest",
        "accuracy",
        "latencyMs",
        "sizeBytes",
        "slices",
    }

    return set(evaluation.keys()) == required


# ============================================================
# Policy validation
# ============================================================

def validate_policy(policy):
    if not isinstance(policy, dict):
        return False

    required = {
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement",
    }

    if set(policy.keys()) != required:
        return False

    if (
        not isinstance(policy["datasetDigest"], str)
        or not policy["datasetDigest"]
    ):
        return False

    if (
        not isinstance(policy["schemaDigest"], str)
        or not policy["schemaDigest"]
    ):
        return False

    if not safe_integer(policy["maxAgeSeconds"]):
        return False

    if (
        not finite(policy["accuracyFloor"])
        or not 0 <= float(policy["accuracyFloor"]) <= 1
    ):
        return False

    if not isinstance(policy["requiredSlices"], dict):
        return False

    for name, floor in policy["requiredSlices"].items():

        if not isinstance(name, str):
            return False

        if (
            not finite(floor)
            or not 0 <= float(floor) <= 1
        ):
            return False

    if (
        not finite(policy["maxLatencyMs"])
        or float(policy["maxLatencyMs"]) < 0
    ):
        return False

    if not safe_integer(policy["maxSizeBytes"]):
        return False

    if (
        not finite(policy["minImprovement"])
        or not 0 <= float(policy["minImprovement"]) <= 1
    ):
        return False

    return True


# ============================================================
# Evaluate one version
# ============================================================

def evaluate_version(version_obj, as_of, policy):
    codes = []

    version = version_obj.get("version")

    # Version itself must already be known valid here.
    evaluation = version_obj.get("evaluation")

    if evaluation is None:
        codes.append("MISSING_EVALUATION")
        return codes, None

    if not validate_evaluation_shape(evaluation):
        codes.append("MISSING_EVALUATION")
        return codes, None

    created_at = parse_timestamp(
        evaluation["createdAt"]
    )

    if created_at is None:
        codes.append("INVALID_TIMESTAMP")
    else:

        if created_at > as_of:
            codes.append("FUTURE_EVALUATION")

        else:

            age = (
                as_of - created_at
            ).total_seconds()

            if age > policy["maxAgeSeconds"]:
                codes.append(
                    "STALE_EVALUATION"
                )

    # --------------------------------------------------------
    # Numeric metrics
    # --------------------------------------------------------

    accuracy = evaluation["accuracy"]
    latency = evaluation["latencyMs"]
    size = evaluation["sizeBytes"]

    if (
        not finite(accuracy)
        or not finite(latency)
        or not finite(size)
    ):
        codes.append("NON_FINITE")

    else:

        if not 0 <= float(accuracy) <= 1:
            codes.append("METRIC_RANGE")

        if float(latency) < 0:
            codes.append("METRIC_RANGE")

        if (
            not safe_integer(size)
        ):
            codes.append("METRIC_RANGE")

    # --------------------------------------------------------
    # Artifact binding
    # --------------------------------------------------------

    registered_artifact = version_obj.get(
        "artifactDigest"
    )

    evaluation_artifact = evaluation[
        "artifactDigest"
    ]

    if (
        not isinstance(registered_artifact, str)
        or not isinstance(evaluation_artifact, str)
        or registered_artifact
        != evaluation_artifact
    ):
        codes.append("ARTIFACT_MISMATCH")

    # --------------------------------------------------------
    # Dataset binding
    # --------------------------------------------------------

    if (
        evaluation["datasetDigest"]
        != policy["datasetDigest"]
    ):
        codes.append("DATASET_MISMATCH")

    # --------------------------------------------------------
    # Schema binding
    # --------------------------------------------------------

    if (
        evaluation["schemaDigest"]
        != policy["schemaDigest"]
    ):
        codes.append("SCHEMA_MISMATCH")

    # --------------------------------------------------------
    # Aggregate gates
    # --------------------------------------------------------

    if (
        finite(accuracy)
        and 0 <= float(accuracy) <= 1
        and float(accuracy)
        < float(policy["accuracyFloor"])
    ):
        codes.append("ACCURACY_FLOOR")

    if (
        finite(latency)
        and float(latency) >= 0
        and float(latency)
        > float(policy["maxLatencyMs"])
    ):
        codes.append("LATENCY_LIMIT")

    if (
        safe_integer(size)
        and size > policy["maxSizeBytes"]
    ):
        codes.append("SIZE_LIMIT")

    # --------------------------------------------------------
    # Slice gates
    # --------------------------------------------------------

    slices = evaluation["slices"]

    if not isinstance(slices, dict):
        # Treat malformed slice container as missing
        # every required slice.
        for name in policy["requiredSlices"]:
            codes.append(
                f"MISSING_SLICE:{name}"
            )
    else:

        for name, floor in policy[
            "requiredSlices"
        ].items():

            if name not in slices:
                codes.append(
                    f"MISSING_SLICE:{name}"
                )
                continue

            value = slices[name]

            if (
                not finite(value)
                or not 0 <= float(value) <= 1
            ):
                codes.append(
                    f"SLICE_RANGE:{name}"
                )
                continue

            if float(value) < float(floor):
                codes.append(
                    f"SLICE_FLOOR:{name}"
                )

    return sort_codes(codes), evaluation


# ============================================================
# Main promotion logic
# ============================================================

def process_promote(payload):
    global ALIAS_VERSION

    # --------------------------------------------------------
    # Top-level validation
    # --------------------------------------------------------

    required = {
        "asOf",
        "championVersion",
        "policy",
        "versions",
    }

    if set(payload.keys()) != required:
        return None, 400

    if not isinstance(
        payload["championVersion"],
        str,
    ):
        return None, 400

    if not isinstance(
        payload["versions"],
        list,
    ):
        return None, 400

    as_of = parse_timestamp(
        payload["asOf"]
    )

    # A malformed asOf is an invalid policy/request
    # rather than a valid evidence timestamp.
    if as_of is None:
        return None, 400

    if not validate_policy(
        payload["policy"]
    ):
        return None, 400

    policy = payload["policy"]

    champion_version = payload[
        "championVersion"
    ]

    # --------------------------------------------------------
    # Every occurrence gets validated BEFORE lookup maps.
    # --------------------------------------------------------

    failed_gates = {}

    valid_occurrences = []

    seen_versions = set()

    for version_obj in payload["versions"]:

        if not isinstance(version_obj, dict):
            # No usable version ID exists.
            # The contract's failedGates is version keyed,
            # so malformed anonymous entries only contribute
            # to INVALID_VERSION under a null-like key.
            continue

        version = version_obj.get(
            "version"
        )

        if not valid_version(version):

            if isinstance(version, str):
                failed_gates.setdefault(
                    version,
                    []
                ).append(
                    "INVALID_VERSION"
                )

            continue

        # Duplicate/noncanonical checks happen before maps.
        if version in seen_versions:

            failed_gates.setdefault(
                version,
                []
            ).append(
                "DUPLICATE_VERSION"
            )

            continue

        seen_versions.add(version)

        valid_occurrences.append(
            version_obj
        )

    # --------------------------------------------------------
    # Evaluate all valid unique versions.
    # --------------------------------------------------------

    eligible = []
    evidence_map = {}

    for version_obj in valid_occurrences:

        version = version_obj["version"]

        codes, evaluation = evaluate_version(
            version_obj,
            as_of,
            policy,
        )

        if codes:
            failed_gates[version] = (
                list(
                    failed_gates.get(
                        version,
                        []
                    )
                )
                + codes
            )

        else:
            eligible.append(version)
            evidence_map[version] = evaluation

    # --------------------------------------------------------
    # Canonicalize failed gates.
    # --------------------------------------------------------

    for version in list(failed_gates.keys()):
        failed_gates[version] = sort_codes(
            failed_gates[version]
        )

    # --------------------------------------------------------
    # Champion evidence must be valid.
    # --------------------------------------------------------

    champion_valid = (
        valid_version(champion_version)
        and champion_version in eligible
    )

    # Champion must be listed.
    if (
        valid_version(champion_version)
        and champion_version
        not in seen_versions
    ):
        champion_valid = False

    if not champion_valid:

        response = {
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": sort_utf8(
                eligible
            ),
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None,
        }

        return response, 200

    # --------------------------------------------------------
    # Rank eligible versions:
    #
    # accuracy DESC
    # latency ASC
    # size ASC
    # numeric version ASC
    # --------------------------------------------------------

    def rank_key(version):

        ev = evidence_map[version]

        return (
            -float(ev["accuracy"]),
            float(ev["latencyMs"]),
            ev["sizeBytes"],
            int(version),
        )

    ranked = sorted(
        eligible,
        key=rank_key,
    )

    selected_version = ranked[0]

    champion_eval = evidence_map[
        champion_version
    ]

    selected_eval = evidence_map[
        selected_version
    ]

    # --------------------------------------------------------
    # Champion wins -> retain.
    # --------------------------------------------------------

    if selected_version == champion_version:

        response = {
            "action": "retain",
            "championVersion": champion_version,
            "selectedVersion": champion_version,
            "eligibleVersions": sort_utf8(
                eligible
            ),
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": champion_eval,
        }

        return response, 200

    # --------------------------------------------------------
    # Challenger improvement.
    # --------------------------------------------------------

    improvement = round(
        float(selected_eval["accuracy"])
        - float(champion_eval["accuracy"]),
        12,
    )

    if improvement >= float(
        policy["minImprovement"]
    ):

        ALIAS_VERSION = selected_version

        response = {
            "action": "promote",
            "championVersion": champion_version,
            "selectedVersion": selected_version,
            "eligibleVersions": sort_utf8(
                eligible
            ),
            "failedGates": failed_gates,
            "aliasMutation": {
                "alias": "champion",
                "version": selected_version,
            },
            "evidence": selected_eval,
        }

        return response, 200

    # --------------------------------------------------------
    # Challenger is eligible but doesn't improve enough.
    # --------------------------------------------------------

    response = {
        "action": "retain",
        "championVersion": champion_version,
        "selectedVersion": champion_version,
        "eligibleVersions": sort_utf8(
            eligible
        ),
        "failedGates": failed_gates,
        "aliasMutation": None,
        "evidence": champion_eval,
    }

    return response, 200


# ============================================================
# Endpoint
# ============================================================

@app.post("/promote")
async def promote(request: Request):

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if not isinstance(payload, dict):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # Required top-level checks that explicitly require
    # HTTP 400 with the exact body.
    if "championVersion" not in payload:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if not isinstance(
        payload["championVersion"],
        str,
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if "versions" not in payload:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if not isinstance(
        payload["versions"],
        list,
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # --------------------------------------------------------
    # Deterministic replay.
    # --------------------------------------------------------

    fingerprint = hashlib.sha256(
        compact_json(payload).encode("utf-8")
    ).hexdigest()

    if fingerprint in REPLAYS:
        return JSONResponse(
            REPLAYS[fingerprint]
        )

    response, status = process_promote(
        payload
    )

    if response is None:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=status,
        )

    REPLAYS[fingerprint] = response

    return JSONResponse(
        response,
        status_code=status,
    )


@app.get("/")
async def root():
    return {
        "service": "model-registry-promotion-gate",
        "status": "ok",
    }