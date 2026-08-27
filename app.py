import json
import math
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

# Stateful champion alias.
CHAMPION_ALIAS = None

# Request fingerprint -> response.
REPLAY_CACHE = {}

SAFE_INT_MAX = (1 << 53) - 1

TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T"
    r"(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?"
    r"(Z|[+-]\d{2}:\d{2})$"
)

VERSION_RE = re.compile(r"^[1-9][0-9]*$")


# ============================================================
# Helpers
# ============================================================

def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def utf8_key(value):
    return value.encode("utf-8")


def sort_utf8(values):
    return sorted(values, key=utf8_key)


def sort_codes(values):
    return sorted(set(values), key=utf8_key)


def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INT_MAX
    )


def finite_number(value):
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

    m = TIMESTAMP_RE.fullmatch(value)

    if not m:
        return None

    fraction = m.group(5) or ""
    fraction = fraction.ljust(3, "0")

    tz_text = m.group(6)

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
            year=int(m.group(1)[0:4]),
            month=int(m.group(1)[5:7]),
            day=int(m.group(1)[8:10]),
            hour=int(m.group(2)),
            minute=int(m.group(3)),
            second=int(m.group(4)),
            microsecond=int(fraction) * 1000,
            tzinfo=tz,
        )

        return dt.astimezone(timezone.utc)

    except ValueError:
        return None


# ============================================================
# Version
# ============================================================

def valid_version(value):
    if not isinstance(value, str):
        return False

    if VERSION_RE.fullmatch(value) is None:
        return False

    try:
        number = int(value)
    except ValueError:
        return False

    return 1 <= number <= SAFE_INT_MAX


# ============================================================
# Policy
# ============================================================

def valid_policy(policy):
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

    if any(key not in policy for key in required):
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
        not finite_number(policy["accuracyFloor"])
        or not 0 <= float(policy["accuracyFloor"]) <= 1
    ):
        return False

    if not isinstance(policy["requiredSlices"], dict):
        return False

    for name, floor in policy["requiredSlices"].items():

        if not isinstance(name, str):
            return False

        if (
            not finite_number(floor)
            or not 0 <= float(floor) <= 1
        ):
            return False

    if (
        not finite_number(policy["maxLatencyMs"])
        or float(policy["maxLatencyMs"]) < 0
    ):
        return False

    if not safe_integer(policy["maxSizeBytes"]):
        return False

    if (
        not finite_number(policy["minImprovement"])
        or not 0 <= float(policy["minImprovement"]) <= 1
    ):
        return False

    return True


# ============================================================
# Version evaluation
# ============================================================

def check_version(version_obj, as_of, policy):
    codes = []

    evaluation = version_obj.get("evaluation")

    if evaluation is None:
        return ["MISSING_EVALUATION"], None

    if not isinstance(evaluation, dict):
        return ["MISSING_EVALUATION"], None

    evaluation_required = [
        "createdAt",
        "artifactDigest",
        "datasetDigest",
        "schemaDigest",
        "accuracy",
        "latencyMs",
        "sizeBytes",
        "slices",
    ]

    if any(
        key not in evaluation
        for key in evaluation_required
    ):
        codes.append("MISSING_EVALUATION")
        return sort_codes(codes), evaluation

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

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
    # Numeric evidence
    # --------------------------------------------------------

    accuracy = evaluation["accuracy"]
    latency = evaluation["latencyMs"]
    size = evaluation["sizeBytes"]

    if (
        not finite_number(accuracy)
        or not finite_number(latency)
        or not safe_integer(size)
    ):
        codes.append("NON_FINITE")

    else:

        if not 0 <= float(accuracy) <= 1:
            codes.append("METRIC_RANGE")

        if float(latency) < 0:
            codes.append("METRIC_RANGE")

    # --------------------------------------------------------
    # Artifact lineage
    # --------------------------------------------------------

    if (
        not isinstance(
            version_obj.get("artifactDigest"),
            str,
        )
        or not isinstance(
            evaluation["artifactDigest"],
            str,
        )
        or version_obj["artifactDigest"]
        != evaluation["artifactDigest"]
    ):
        codes.append("ARTIFACT_MISMATCH")

    # --------------------------------------------------------
    # Dataset lineage
    # --------------------------------------------------------

    if (
        evaluation["datasetDigest"]
        != policy["datasetDigest"]
    ):
        codes.append("DATASET_MISMATCH")

    # --------------------------------------------------------
    # Schema lineage
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
        finite_number(accuracy)
        and 0 <= float(accuracy) <= 1
        and float(accuracy)
        < float(policy["accuracyFloor"])
    ):
        codes.append("ACCURACY_FLOOR")

    if (
        finite_number(latency)
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
    # Required slices
    # --------------------------------------------------------

    slices = evaluation["slices"]

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

            if not finite_number(value):
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
# Promotion
# ============================================================

def build_promotion(payload):
    global CHAMPION_ALIAS

    # --------------------------------------------------------
    # Required request fields
    # --------------------------------------------------------

    required = [
        "asOf",
        "championVersion",
        "policy",
        "versions",
    ]

    if any(key not in payload for key in required):
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
        return None

    policy = payload["policy"]

    if not valid_policy(policy):
        return None

    champion = payload[
        "championVersion"
    ]

    failed_gates = {}
    eligible = []
    evidence = {}

    seen = set()

    # --------------------------------------------------------
    # Validate every occurrence before lookup maps.
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

        codes, ev = check_version(
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

    # --------------------------------------------------------
    # Deduplicate / sort codes.
    # --------------------------------------------------------

    for version in failed_gates:
        failed_gates[version] = sort_codes(
            failed_gates[version]
        )

    eligible = sort_utf8(eligible)

    # --------------------------------------------------------
    # Champion must be valid and eligible.
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
    # --------------------------------------------------------

    def ranking(version):

        ev = evidence[version]

        return (
            -float(ev["accuracy"]),
            float(ev["latencyMs"]),
            ev["sizeBytes"],
            int(version),
        )

    ranked = sorted(
        eligible,
        key=ranking,
    )

    challenger = ranked[0]

    # Champion remains winner.
    if challenger == champion:

        return {
            "action": "retain",
            "championVersion": champion,
            "selectedVersion": champion,
            "eligibleVersions": eligible,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": evidence[champion],
        }

    # --------------------------------------------------------
    # Improvement gate.
    # --------------------------------------------------------

    improvement = round(
        float(evidence[challenger]["accuracy"])
        - float(evidence[champion]["accuracy"]),
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
            "evidence": evidence[champion],
        }

    # --------------------------------------------------------
    # Promote.
    # --------------------------------------------------------

    CHAMPION_ALIAS = challenger

    return {
        "action": "promote",
        "championVersion": champion,
        "selectedVersion": challenger,
        "eligibleVersions": eligible,
        "failedGates": failed_gates,
        "aliasMutation": {
            "alias": "champion",
            "version": challenger,
        },
        "evidence": evidence[challenger],
    }


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

    # These cases are explicitly specified as HTTP 400.
    if (
        "championVersion" not in payload
        or not isinstance(
            payload["championVersion"],
            str,
        )
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    if (
        "versions" not in payload
        or not isinstance(
            payload["versions"],
            list,
        )
    ):
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    # --------------------------------------------------------
    # Replay identity.
    # --------------------------------------------------------

    request_key = compact_json(payload)

    if request_key in REPLAY_CACHE:
        return JSONResponse(
            REPLAY_CACHE[request_key]
        )

    result = build_promotion(payload)

    if result is None:
        return JSONResponse(
            {"error": "INVALID_INPUT"},
            status_code=400,
        )

    REPLAY_CACHE[request_key] = result

    return JSONResponse(result)


@app.get("/")
async def root():
    return {
        "service": "model-registry-promotion-gate",
        "status": "ok",
    }
