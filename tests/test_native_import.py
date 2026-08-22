import hashlib
import json
from pathlib import Path

import chess_rl_native


EXPECTED_COMMIT = "53e6a841dcda7059a2af363d85f785ef1817304a"
EXPECTED_HEADER_SHA256 = "f2c8e2e929641e2c71cbe9d8abd718cf3cac46c2a34531215ebd733905e98d7f"
EXPECTED_LICENSE_SHA256 = "5860e7607afbf4c7e91bad5549b71d16fc4eceb90f0c671cd77a343ae7461a2a"


def test_native_metadata_and_vendor_provenance():
    assert isinstance(chess_rl_native.__version__, str)
    assert chess_rl_native.native_abi_version() == "1"
    assert chess_rl_native.chess_library_commit() == EXPECTED_COMMIT
    assert (
        chess_rl_native.chess_library_header_sha256() == EXPECTED_HEADER_SHA256
    )

    build_info = chess_rl_native.build_info()
    assert build_info["cxx_standard"] == "c++17"
    assert build_info["cplusplus"] == 201703
    assert build_info["chess_library_commit"] == EXPECTED_COMMIT
    assert build_info["native_abi_version"] == "1"

    package_dir = Path(chess_rl_native.__file__).resolve().parent
    package_license_path = package_dir / "LICENSE"
    provenance_path = package_dir / "third_party" / "chess-library" / "PROVENANCE.json"
    license_path = package_dir / "third_party" / "chess-library" / "LICENSE"
    header_path = package_dir / "third_party" / "chess-library" / "include" / "chess.hpp"
    provenance = json.loads(provenance_path.read_text())

    assert provenance["commit"] == EXPECTED_COMMIT
    assert provenance["include/chess.hpp_sha256"] == EXPECTED_HEADER_SHA256
    assert provenance["LICENSE_sha256"] == EXPECTED_LICENSE_SHA256
    assert "MIT License" in package_license_path.read_text()
    assert hashlib.sha256(header_path.read_bytes()).hexdigest() == EXPECTED_HEADER_SHA256
    assert hashlib.sha256(license_path.read_bytes()).hexdigest() == EXPECTED_LICENSE_SHA256
