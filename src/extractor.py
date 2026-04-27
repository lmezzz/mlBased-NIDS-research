import numpy as np
import pandas as pd

def feature_control_kdd(df: pd.DataFrame) -> pd.DataFrame:
    features = [
    "duration",
    "protocol",
    "src_bytes",
    "dst_bytes",
    "bytes_per_sec",
    "byte_ratio",
    "label"
    ]

    df = df.copy()

    # Rename columns to unified schema
    df = df.rename(columns={
        "protocol_type": "protocol"
    })

    # Ensure numeric
    df["duration"] = df["duration"].astype(float)
    df["src_bytes"] = df["src_bytes"].astype(float)
    df["dst_bytes"] = df["dst_bytes"].astype(float)
    df["label"] = df["class"].copy()  # Keep original for label encoding later


    # Feature engineering
    df["bytes_per_sec"] = (df["src_bytes"] + df["dst_bytes"]) / (df["duration"] + 1e-6)
    df["bytes_per_sec"] = df["bytes_per_sec"].replace([np.inf, -np.inf], 0)

    df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["byte_ratio"] = df["byte_ratio"].replace([np.inf, -np.inf], 0)

    print(f"[Extractor] KDD feature control applied. Resulting shape: {df.shape}")
    print(f"[Extractor] Sample features:\n{df[features].head(3)}")

    return df[[
        "duration",
        "protocol",
        "src_bytes",
        "dst_bytes",
        "bytes_per_sec",
        "byte_ratio",
        "label"
    ]]

def feature_control_cicids(df: pd.DataFrame) -> pd.DataFrame:
    features = [
    "duration",
    "protocol",
    "src_bytes",
    "dst_bytes",
    "bytes_per_sec",
    "byte_ratio",
    "label" 
    ]

    df = df.copy()

    # Rename to match schema
    df = df.rename(columns={
        "Flow Duration": "duration",
        "Protocol": "protocol",
        "Total Length of Fwd Packets": "src_bytes",
        "Total Length of Bwd Packets": "dst_bytes"
    })

    # Convert duration (CICIDS is usually in microseconds)
    df["duration"] = df["duration"] / 1e6

    # Ensure numeric
    df["src_bytes"] = df["src_bytes"].astype(float)
    df["dst_bytes"] = df["dst_bytes"].astype(float)
    df["label"] = df["Label"].copy()  # Keep original for label encoding later

    #converting the protocol int to string to make it same as KDD
    protocol_map = {6: "tcp", 17: "udp", 1: "icmp"}
    df["protocol"] = df["protocol"].map(protocol_map)

    # Feature engineering
    df["bytes_per_sec"] = (df["src_bytes"] + df["dst_bytes"]) / (df["duration"] + 1e-6)
    df["bytes_per_sec"] = df["bytes_per_sec"].replace([np.inf, -np.inf], 0).fillna(0)
    
    df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    df["byte_ratio"] = df["byte_ratio"].replace([np.inf, -np.inf], 0)

    print(f"[Extractor] CICIDS feature control applied. Resulting shape: {df.shape}")
    print(f"[Extractor] Sample features:\n{df[features].head(3)}")

    return df[[
        "duration",
        "protocol",
        "src_bytes",
        "dst_bytes",
        "bytes_per_sec",
        "byte_ratio",
        "label"
    ]]


# ── PORT BUCKET MAPPING ──
def port_bucket(port):
    try:
        port = int(port)
    except:
        return "unknown"
 
    # Named protocol ports — directly encodes which grammar applies
    if port == 80 or port == 8080:   return "http"
    elif port == 443:                 return "https"
    elif port == 22:                  return "ssh"
    elif port == 21 or port == 20:    return "ftp"
    elif port == 53:                  return "dns"
    elif port == 25:                  return "smtp"
    elif port == 23:                  return "telnet"
    elif port == 3306:                return "mysql"
    elif port == 3389:                return "rdp"
 
    # IANA tier fallback — protocol standard classification
    elif port < 1024:                 return "system"
    elif port < 49152:                return "user"
    else:                             return "dynamic"
 
 
# ── SERVICE TO PORT BUCKET (KDD) ──
SERVICE_BUCKET_MAP = {
    "http":        "http",
    "https":       "https",
    "http_443":    "https",
    "ftp":         "ftp",
    "ftp_data":    "ftp",
    "ssh":         "ssh",
    "smtp":        "smtp",
    "domain":      "dns",
    "domain_u":    "dns",
    "telnet":      "telnet",
    "pop_3":       "system",
    "imap4":       "system",
    "finger":      "system",
    "sunrpc":      "system",
    "mtp":         "system",
    "eco_i":       "system",
    "ecr_i":       "system",
    "nntp":        "system",
    "IRC":         "user",
    "X11":         "user",
    "Z39_50":      "user",
    "aol":         "user",
    "auth":        "system",
    "bgp":         "system",
    "courier":     "user",
    "csnet_ns":    "system",
    "ctf":         "user",
    "daytime":     "system",
    "discard":     "system",
    "exec":        "system",
    "gopher":      "system",
    "harvest":     "user",
    "hostnames":   "system",
    "http_2784":   "user",
    "http_8001":   "user",
    "imap4":       "system",
    "iso_tsap":    "system",
    "klogin":      "system",
    "kshell":      "system",
    "ldap":        "system",
    "link":        "user",
    "login":       "system",
    "lycos":       "user",
    "netbios_dgm": "system",
    "netbios_ns":  "system",
    "netbios_ssn": "system",
    "netstat":     "system",
    "ntp_u":       "system",
    "other":       "dynamic",
    "pm_dump":     "user",
    "pop_2":       "system",
    "printer":     "system",
    "private":     "dynamic",
    "red_i":       "system",
    "remote_job":  "system",
    "rje":         "system",
    "shell":       "system",
    "sql_net":     "user",
    "supdup":      "system",
    "systat":      "system",
    "tftp_u":      "system",
    "tim_i":       "system",
    "time":        "system",
    "urh_i":       "system",
    "urp_i":       "system",
    "uucp":        "system",
    "uucp_path":   "system",
    "vmnet":       "user",
    "whois":       "system",
}
 
def map_service_to_bucket(service):
    return SERVICE_BUCKET_MAP.get(str(service).strip(), "user")
 
 
# ════════════════════════════════════════
# PROTOCOL AWARE FEATURE EXTRACTORS
# ════════════════════════════════════════
 
def feature_protocol_aware_kdd(df):
    """
    Extracts protocol-aware features from NSL-KDD.
    Returns ALL protocol-aware features available on KDD side.
    Aligner will select which ones to use per experiment.
    """
    out = df.copy()
 
    # syn_ratio → serror_rate is already a SYN error rate (0.0 to 1.0)
    # Maps to: proportion of connections with SYN errors (half-open)
    out["syn_ratio"] = df["serror_rate"].astype(float)
 
    # rst_ratio → rerror_rate is REJ error rate (0.0 to 1.0)
    # Maps to: proportion of connections rejected (RST responses)
    out["rst_ratio"] = df["rerror_rate"].astype(float)
 
    # fin_ratio → 1 if flag is SF (connection completed via FIN), else 0
    # SF = SYN + FIN = normal open and close = grammar compliant
    out["fin_ratio"] = (df["flag"] == "SF").astype(float)
 
    # data_pkt_ratio → 1 if src_bytes > 0 (actual payload sent), else 0
    # Binary proxy: did this connection carry real data?
    out["data_pkt_ratio"] = (df["src_bytes"].astype(float) > 0).astype(float)
 
    # service_bucket → maps KDD service name to IANA protocol bucket
    out["service_bucket"] = df["service"].apply(map_service_to_bucket)
 
    # NOTE: window_ratio and min_seg_size have no KDD equivalent
    # They are extracted on CICIDS side only
    # Aligner handles the asymmetry per experiment
 
    protocol_aware_cols = [
        "syn_ratio", "rst_ratio", "fin_ratio",
        "data_pkt_ratio", "service_bucket"
    ]
 
    print(f"[Extractor] KDD protocol-aware features: {protocol_aware_cols}")
    print(f"[Extractor] Shape: {out[protocol_aware_cols].shape}")
    return out[protocol_aware_cols]
 
 
def feature_protocol_aware_cicids(df):
    """
    Extracts protocol-aware features from CICIDS.
    Returns ALL protocol-aware features including CICIDS-only ones.
    Aligner will select which ones to use per experiment.
    """
    out = df.copy()
 
    total_fwd = df["Total Fwd Packets"].astype(float).replace(0, np.nan)
 
    # syn_ratio → SYN flags as proportion of forward packets
    # High value = SYN flood signature
    out["syn_ratio"] = (
        df["SYN Flag Count"].astype(float) / total_fwd
    ).fillna(0).clip(0, 1)
 
    # rst_ratio → RST flags as proportion of forward packets
    # High value = port scan (closed ports respond with RST)
    out["rst_ratio"] = (
        df["RST Flag Count"].astype(float) / total_fwd
    ).fillna(0).clip(0, 1)
 
    # fin_ratio → FIN flags as proportion of forward packets
    # Abnormal high = FIN scan
    # Abnormal zero = Slowloris (connections never close)
    out["fin_ratio"] = (
        df["FIN Flag Count"].astype(float) / total_fwd
    ).fillna(0).clip(0, 1)
 
    # data_pkt_ratio → actual data packets as proportion of forward packets
    # Low value = SYN/probe traffic with no payload
    # High value = data-carrying traffic (normal or HTTP flood)
    out["data_pkt_ratio"] = (
        df["act_data_pkt_fwd"].astype(float) / total_fwd
    ).fillna(0).clip(0, 1)
 
    # window_ratio → client/server initial window size asymmetry
    # Server window shrinks under load → ratio spikes
    # Scanner tools use non-standard window sizes → ratio anomalous
 
    # service_bucket → maps destination port to IANA protocol bucket
    out["service_bucket"] = df["Destination Port"].apply(port_bucket)
 
    protocol_aware_cols = [
        "syn_ratio", "rst_ratio", "fin_ratio", "data_pkt_ratio",
        "service_bucket"
    ]
 
    print(f"[Extractor] CICIDS protocol-aware features: {protocol_aware_cols}")
    print(f"[Extractor] Shape: {out[protocol_aware_cols].shape}")
    return out[protocol_aware_cols]
 
