from modelgate.fingerprint import sha256_bytes, sha256_file


def test_known_digest_and_content_changes(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(path) == sha256_bytes(b"abc") == expected
    path.write_bytes(b"abc\n")
    assert sha256_file(path) != expected
