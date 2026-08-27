import json
import math
import re
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

SAFE_MAX = (1 << 53) - 1

# Stateful alias.
# The service process retains this across requests.
ALIAS_VERSION = None

# Exact request -> response replay cache.
REPLAY_CACHE = {}

TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

VERSION_RE = re.compile(r"^[1-9][0-9]*$")


# ============================================================
# Deterministic helpers
# ============================================================

def compact(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8(value):
    return value.encode("utf-8")


def sort_utf8(values):
    return sorted(values, key=utf8)


def sort_codes(values):
    return sorted(set(values), key=utf8)


def is_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_MAX
    )


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


# ============================================================
# Timestamp
# ============================================================

def parse_timestamp(value):
    if not isinstance(value, str):
        return None

    match = TIMESTAMP_RE.fullmatch(value)

    if not match:
        return None

    fraction = match.group(5) or ""
    fraction = fraction.ljust(3, "0")

    tz_text = match.group(6)

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

        dt = datetime(
            int(match.group(1)[0:4]),
            int(match.group(1)[5:7]),
            int(match.group(1)[8:10]),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(fraction) * 1000,
            tzinfo=tz,
        )

        return dt.astimezone(timezone.utc)

    except (ValueError, OverflowError):
        return None


# ============================================================
# Version
# ============================================================

def valid_version(version):
    if not isinstance(version, str):
        return False

    if VERSION_RE.fullmatch(version) is None:
        return False

    try:
        n = int(version)
    except ValueError:
        return False

    return 1 <= n <= SAFE_MAX


# ============================================================
# Policy validation
# ============================================================

def policy_is_valid(policy):
    if not isinstance(policy, dict):
        return False

    required = [
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement",
    ]

    if any(k not in policy for k in required):
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

    if not is_safe_integer(policy["maxAgeSeconds"]):
        return False

    if (
        not is_finite_number(policy["accuracyFloor"])
        or not 0 <= float(policy["accuracyFloor"]) <= 1
    ):
        return False

    if not isinstance(policy["requiredSlices"], dict):
        return False

    for name, floor in policy["requiredSlices"].items():
        if not isinstance(name, str):
            return False

        if (
            not is_finite_number(floor)
            or not 0 <= float(floor) <= 1
        ):
            return False

    if (
        not is_finite_number(policy["maxLatencyMs"])
        or float(policy["maxLatencyMs"]) < 0
    ):
        return False

    if not is_safe_integer(policy["maxSizeBytes"]):
        return False

    if (
        not is_finite_number(policy["minImprovement"])
        or not 0 <= float(policy["minImprovement"]) <= 1
    ):
        return False

    return True


# ============================================================
# Version gate evaluation
# ============================================================

def evaluate_version(version_obj, as_of, policy):
    codes = []

    evaluation = version_obj.get("evaluation")

    if not isinstance(evaluation, dict):
        return ["MISSING_EVALUATION"], None

    required = [
        "createdAt",
        "artifactDigest",
        "datasetDigest",
        "schemaDigest",
        "accuracy",
        "latencyMs",
        "sizeBytes",
        "slices",
    ]

    if any(k not in evaluation for k in required):
        return ["MISSING_EVALUATION"], evaluation

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    created = parse_timestamp(
        evaluation["createdAt"]
    )

    if created is None:
        codes.append("INVALID_TIMESTAMP")
    else:
        if created > as_of:
            codes.append("FUTURE_EVALUATION")
        else:
            age = (
                as_of - created
            ).total_seconds()

            if age > policy["maxAgeSeconds"]:
                codes.append("STALE_EVALUATION")

    # --------------------------------------------------------
    # Numeric evidence
    # --------------------------------------------------------

    accuracy = evaluation["accuracy"]
    latency = evaluation["latencyMs"]
    size = evaluation["sizeBytes"]

    accuracy_valid = is_finite_number(accuracy)
    latency_valid = is_finite_number(latency)
    size_valid = is_safe_integer(size)

    if not accuracy_valid or not latency_valid or not size_valid:
        codes.append("NON_FINITE")

    if accuracy_valid:
        if not 0 <= float(accuracy) <= 1:
            codes.append("METRIC_RANGE")

    if latency_valid:
        if float(latency) < 0:
            codes.append("METRIC_RANGE")

    if not size_valid:
        codes.append("METRIC_RANGE")

    # --------------------------------------------------------
    # Artifact binding
    # --------------------------------------------------------

    registered_artifact = version_obj.get(
        "artifactDigest"
    )

    evaluated_artifact = evaluation.get(
        "artifactDigest"
    )

    if (
        not isinstance(registered_artifact, str)
        or not isinstance(evaluated_artifact, str)
        or registered_artifact != evaluated_artifact
    ):
        codes.append("ARTIFACT_MISMATCH")

    # --------------------------------------------------------
    # Dataset binding
    # --------------------------------------------------------

    if (
        evaluation.get("datasetDigest")
        != policy["datasetDigest"]
    ):
        codes.append("DATASET_MISMATCH")

    # --------------------------------------------------------
    # Schema binding
    # --------------------------------------------------------

    if (
        evaluation.get("schemaDigest")
        != policy["schemaDigest"]
    ):
        codes.append("SCHEMA_MISMATCH")

    # --------------------------------------------------------
    # Aggregate gates
    # --------------------------------------------------------

    if (
        accuracy_valid
        and 0 <= float(accuracy) <= 1
        and float(accuracy)
        < float(policy["accuracyFloor"])
    ):
        codes.append("ACCURACY_FLOOR")

    if (
        latency_valid
        and float(latency) >= 0
        and float(latency)
        > float(policy["maxLatencyMs"])
    ):
        codes.append("LATENCY_LIMIT")

    if (
        size_valid
        and size > policy["maxSizeBytes"]
    ):
        codes.append("SIZE_LIMIT")

    # --------------------------------------------------------
    # Slice gates
    # --------------------------------------------------------

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):

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

            if not is_finite_number(value):
                codes.append(
                    f"SLICE_RANGE:{name}"
                )
                continue

            if not 0 <= float(value) <= 1:
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

def promote_logic(payload):

    global ALIAS_VERSION

    # Required fields.
    required = [
        "asOf",
        "championVersion",
        "policy",
        "versions",
    ]

    if any(k not in payload for k in required):
        return None

    if not isinstance(
        payload["championVersion"],
        str,
    ):
        return None

    if not isinstance(
        payload["versions"],
        list,
    ):
        return None

    as_of = parse_timestamp(
        payload["asOf"]
    )

    if as_of is None:
        # Invalid request timestamp is request-level invalid input.
        return None

    policy = payload["policy"]

    # IMPORTANT:
    # Policy being present but invalid is NOT HTTP 400.
    # It becomes INVALID_POLICY in failedGates.
    policy_valid = policy_is_valid(policy)

    if not policy_valid:

        # We still need a deterministic response.
        # Every listed valid version receives INVALID_POLICY.
        failed = {}

        seen = set()

        for item in payload["versions"]:

            if not isinstance(item, dict):
                continue

            version = item.get("version")

            if not valid_version(version):
                if isinstance(version, str):
                    failed.setdefault(
                        version,
                        []
                    ).append(
                        "INVALID_VERSION"
                    )
                continue

            if version in seen:
                failed.setdefault(
                    version,
                    []
                ).append(
                    "DUPLICATE_VERSION"
                )
                continue

            seen.add(version)

            failed[version] = [
                "INVALID_POLICY"
            ]

        for version in failed:
            failed[version] = sort_codes(
                failed[version]
            )

        return {
            "action": "block",
            "championVersion": payload[
                "championVersion"
            ],
            "selectedVersion": None,
            "eligibleVersions": [],
            "failedGates": failed,
            "aliasMutation": None,
            "evidence": None,
        }

    failed_gates = {}
    eligible = []
    evidence = {}

    seen = set()

    # --------------------------------------------------------
    # IMPORTANT:
    # Validate duplicate/noncanonical versions BEFORE
    # constructing lookup maps.
    # --------------------------------------------------------

    for item in payload["versions"]:

        if not isinstance(item, dict):
            continue

        version = item.get("version")

        if not valid_version(version):

            if isinstance(version, str):
                failed_gates.setdefault(
                    version,
                    []
                ).append(
                    "INVALID_VERSION"
                )

            continue

        if version in seen:

            failed_gates.setdefault(
                version,
                []
            ).append(
                "DUPLICATE_VERSION"
            )

            continue

        seen.add(version)

        codes, ev = evaluate_version(
            item,
            as_of,
            policy,
        )

        if codes:
            failed_gates[version] = (
                failed_gates.get(
                    version,
                    []
                )
                + codes
            )
        else:
            eligible.append(version)
            evidence[version] = ev

    # Sort/deduplicate codes.
    for version in failed_gates:
        failed_gates[version] = sort_codes(
            failed_gates[version]
        )

    eligible = sort_utf8(eligible)

    champion = payload[
        "championVersion"
    ]

    # --------------------------------------------------------
    # Invalid champion evidence -> block.
    # --------------------------------------------------------

    champion_valid = (
        valid_version(champion)
        and champion in seen
        and champion in eligible
    )

    if not champion_valid:

        return {
            "action": "block",
            "championVersion": champion,
            "selectedVersion": None,
            "eligibleVersions": eligible,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None,
        }

    # --------------------------------------------------------
    # Rank eligible versions.
    #
    # Accuracy DESC
    # Latency ASC
    # Size ASC
    # Numeric version ASC
    # --------------------------------------------------------

    def rank_key(version):
        ev = evidence[version]

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

    winner = ranked[0]

    champion_ev = evidence[
        champion
    ]

    # --------------------------------------------------------
    # Champion remains best.
    # --------------------------------------------------------

    if winner == champion:

        return {
            "action": "retain",
            "championVersion": champion,
            "selectedVersion": champion,
            "eligibleVersions": eligible,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": champion_ev,
        }

    winner_ev = evidence[
        winner
    ]

    # --------------------------------------------------------
    # Improvement.
    # --------------------------------------------------------

    improvement = round(
        float(winner_ev["accuracy"])
        - float(champion_ev["accuracy"]),
        12,
    )

    if improvement < float(
        policy["minImprovement"]
    ):

        return {
            "action": "retain",
            "championVersion": champion,
            "selectedVersion": champion,
            "eligibleVersions": eligible,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": champion_ev,
        }

    # --------------------------------------------------------
    # Promotion.
    # --------------------------------------------------------

    ALIAS_VERSION = winner

    return {
        "action": "promote",
        "championVersion": champion,
        "selectedVersion": winner,
        "eligibleVersions": eligible,
        "failedGates": failed_gates,
        "aliasMutation": {
            "alias": "champion",
            "version": winner,
        },
        "evidence": winner_ev,
    }


# ============================================================
# POST /promote
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

    # Explicit HTTP-400 contract cases.
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

    if "asOf" not in payload:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if "policy" not in payload:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # Request-level replay.
    replay_key = compact(payload)

    if replay_key in REPLAY_CACHE:
        return JSONResponse(
            REPLAY_CACHE[replay_key]
        )

    result = promote_logic(payload)

    if result is None:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    REPLAY_CACHE[replay_key] = result

    return JSONResponse(result)


@app.get("/")
async def root():
    return {
        "service": "model-registry-promotion-gate",
        "status": "ok",
    }
