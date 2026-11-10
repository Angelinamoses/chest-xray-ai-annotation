
import requests

API_URL = "http://127.0.0.1:8000"


def test_invalid_extension():
    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                "invalid_file.txt",
                b"This is not a DICOM file",
                "text/plain"
            )
        },
        timeout=30
    )

    assert response.status_code == 400, response.text
    print("PASS: Invalid extension rejected with HTTP 400")


def test_empty_file():
    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                "empty.dcm",
                b"",
                "application/dicom"
            )
        },
        timeout=30
    )

    assert response.status_code == 400, response.text
    print("PASS: Empty DICOM rejected with HTTP 400")


def test_corrupted_dicom():
    response = requests.post(
        f"{API_URL}/predict",
        files={
            "file": (
                "corrupted.dcm",
                b"This is deliberately corrupted DICOM content",
                "application/dicom"
            )
        },
        timeout=30
    )

    assert response.status_code == 422, response.text
    print("PASS: Corrupted DICOM rejected with HTTP 422")


if __name__ == "__main__":

    print("\nRunning negative API integration tests...\n")

    test_invalid_extension()
    test_empty_file()
    test_corrupted_dicom()

    print("\n--------------------------------")
    print("ALL NEGATIVE TESTS PASSED!")
    print("--------------------------------")
