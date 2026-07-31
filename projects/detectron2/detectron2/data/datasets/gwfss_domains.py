import os
import re


GWFSS_SOURCE_DOMAINS = (
    "CIMMYT",
    "ETHZ",
    "Inrae",
    "NJAU",
    "RRES",
    "Uliege",
    "UQ",
    "USASK",
    "UTokyo",
)
GWFSS_ALL_DOMAINS = GWFSS_SOURCE_DOMAINS + ("Arvalis",)

_DOMAIN_ALIASES = {
    "INRAE": "Inrae",
    "UQ_new": "UQ",
    "ULiege_CRA-W": "Uliege",
    "ULiege - CRA-W": "Uliege",
}


def infer_gwfss_domain(path):
    """Return the protocol-stable domain id and a readable domain name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"domain([1-9])_", stem, flags=re.IGNORECASE)
    if match:
        domain_id = int(match.group(1)) - 1
        return domain_id, "domain{}".format(domain_id + 1)

    domain_name = os.path.basename(os.path.dirname(path))
    domain_name = _DOMAIN_ALIASES.get(domain_name, domain_name)
    if domain_name in GWFSS_ALL_DOMAINS:
        return GWFSS_ALL_DOMAINS.index(domain_name), domain_name
    raise ValueError("Cannot infer a GWFSS domain from '{}'".format(path))
