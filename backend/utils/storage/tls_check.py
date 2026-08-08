"""
Standalone TLS diagnostic — connects directly to your Neo4j Aura instance's
bolt port using raw ssl/socket, bypassing the neo4j driver entirely, and
prints exactly which certificate it receives.

Run with:  python tls_check.py

If this fails with the same "self-signed certificate in certificate chain"
error, it CONFIRMS something between you and Aura (antivirus, firewall,
corporate VPN) is intercepting the connection — this has nothing to do with
the neo4j Python package, your code, or your credentials.
"""
import socket
import ssl

# Fill in your Aura hostname (from NEO4J_URI, without the neo4j+s:// prefix)
HOST = "61f6bdcd.databases.neo4j.io"
PORT = 7687

print(f"Connecting to {HOST}:{PORT} ...")

ctx = ssl.create_default_context()

try:
    with socket.create_connection((HOST, PORT), timeout=10) as sock:
        with ctx.wrap_socket(sock, server_hostname=HOST) as ssock:
            cert = ssock.getpeercert()
            print("\n✅ TLS handshake succeeded.")
            print("Certificate subject:", cert.get("subject"))
            print("Certificate issuer :", cert.get("issuer"))
except ssl.SSLCertVerificationError as e:
    print("\n❌ TLS handshake FAILED with a certificate verification error:")
    print(f"   {e}")
    print(
        "\nThis confirms something on your machine or network (antivirus "
        "HTTPS scanning, corporate firewall/VPN, or similar) is intercepting "
        "the TLS connection and presenting its own certificate instead of "
        "Neo4j's real one. Try:\n"
        "  1. Disable antivirus HTTPS/SSL scanning (or add a python.exe "
        "exception) and re-run this script.\n"
        "  2. If on a VPN or corporate/school network, switch to a mobile "
        "hotspot and re-run this script.\n"
        "  3. Whichever one fixes it — that's your cause."
    )
except Exception as e:
    print(f"\n❌ Connection failed for a different reason: {type(e).__name__}: {e}")
    print("This looks like a network/firewall block on port 7687 rather than "
          "a certificate issue — check that port 7687 isn't blocked outbound.")