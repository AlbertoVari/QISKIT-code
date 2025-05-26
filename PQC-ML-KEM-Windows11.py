import ctypes
from ctypes import wintypes

# Load the CNG library
bcrypt = ctypes.WinDLL('bcrypt.dll')

# Define constants
BCRYPT_SUCCESS = 0
BCRYPT_ML_KEM_ALGORITHM = "ML-KEM"

# Define structures and prototypes
class BCRYPT_ALG_HANDLE(ctypes.Structure):
    pass

class BCRYPT_KEY_HANDLE(ctypes.Structure):
    pass

bcrypt.BCryptOpenAlgorithmProvider.argtypes = [
    ctypes.POINTER(BCRYPT_ALG_HANDLE),  # phAlgorithm
    wintypes.LPCWSTR,  # pszAlgId
    wintypes.LPCWSTR,  # pszImplementation
    wintypes.DWORD  # dwFlags
]

bcrypt.BCryptOpenAlgorithmProvider.restype = wintypes.ULONG

bcrypt.BCryptGenerateKeyPair.argtypes = [
    BCRYPT_ALG_HANDLE,  # hAlgorithm
    ctypes.POINTER(BCRYPT_KEY_HANDLE),  # phKey
    wintypes.ULONG,  # dwLength
    wintypes.ULONG  # dwFlags
]

bcrypt.BCryptGenerateKeyPair.restype = wintypes.ULONG

bcrypt.BCryptFinalizeKeyPair.argtypes = [
    BCRYPT_KEY_HANDLE,  # hKey
    wintypes.DWORD  # dwFlags
]

bcrypt.BCryptFinalizeKeyPair.restype = wintypes.ULONG

bcrypt.BCryptExportKey.argtypes = [
    BCRYPT_KEY_HANDLE,  # hKey
    BCRYPT_KEY_HANDLE,  # hExportKey
    wintypes.LPCWSTR,  # pszBlobType
    ctypes.POINTER(ctypes.c_ubyte),  # pbOutput
    wintypes.ULONG,  # cbOutput
    ctypes.POINTER(wintypes.ULONG),  # pcbResult
    wintypes.DWORD  # dwFlags
]

bcrypt.BCryptExportKey.restype = wintypes.ULONG

# Open the ML-KEM algorithm provider
def open_algorithm_provider():
    alg_handle = BCRYPT_ALG_HANDLE()
    status = bcrypt.BCryptOpenAlgorithmProvider(
        ctypes.byref(alg_handle),
        BCRYPT_ML_KEM_ALGORITHM,
        None,
        0
    )
    if status != BCRYPT_SUCCESS:
        raise Exception(f"BCryptOpenAlgorithmProvider failed with status {status}")
    return alg_handle

# Generate a key pair
def generate_key_pair(alg_handle):
    key_handle = BCRYPT_KEY_HANDLE()
    status = bcrypt.BCryptGenerateKeyPair(
        alg_handle,
        ctypes.byref(key_handle),
        512,  # Key length
        0
    )
    if status != BCRYPT_SUCCESS:
        raise Exception(f"BCryptGenerateKeyPair failed with status {status}")

    status = bcrypt.BCryptFinalizeKeyPair(key_handle, 0)
    if status != BCRYPT_SUCCESS:
        raise Exception(f"BCryptFinalizeKeyPair failed with status {status}")

    return key_handle

# Export the public key
def export_public_key(key_handle):
    buffer_size = wintypes.ULONG()
    status = bcrypt.BCryptExportKey(
        key_handle,
        None,
        "PUBLICBLOB",
        None,
        0,
        ctypes.byref(buffer_size),
        0
    )
    if status != BCRYPT_SUCCESS:
        raise Exception(f"BCryptExportKey (size) failed with status {status}")

    buffer = (ctypes.c_ubyte * buffer_size.value)()
    status = bcrypt.BCryptExportKey(
        key_handle,
        None,
        "PUBLICBLOB",
        buffer,
        buffer_size.value,
        ctypes.byref(buffer_size),
        0
    )
    if status != BCRYPT_SUCCESS:
        raise Exception(f"BCryptExportKey failed with status {status}")

    return bytes(buffer)

# Main function
def main():
    try:
        alg_handle = open_algorithm_provider()
        key_handle = generate_key_pair(alg_handle)
        public_key = export_public_key(key_handle)
        print("Public Key:", public_key)
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()
