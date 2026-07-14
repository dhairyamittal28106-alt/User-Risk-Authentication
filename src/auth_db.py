import hashlib
import hmac
import os
from pathlib import Path

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError


ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT_DIR / ".env"


def load_env_file() -> None:
    if not ENV_PATH.exists():
        return

    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env_file()

MONGODB_URI = os.environ.get("MONGODB_URI", "")
DB_NAME = os.environ.get("MONGODB_DB_NAME", "RiskShieldCDAC")
USERS_COLLECTION = os.environ.get("MONGODB_USERS_COLLECTION", "users")
AUTH_SECRET = os.environ.get("AUTH_SECRET", "change-this-local-secret")


def get_users_collection():
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not configured. Add it to .env.")

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
    database = client[DB_NAME]
    users = database[USERS_COLLECTION]
    try:
        users.create_index(
            "username",
            unique=True,
            partialFilterExpression={"username": {"$type": "string"}},
        )
    except ServerSelectionTimeoutError as exc:
        raise RuntimeError(
            "MongoDB Atlas rejected the network connection. Add your current IP "
            "address to Atlas Network Access, then try again."
        ) from exc
    return users


def hash_password(password: str) -> str:
    return hashlib.sha256(f"{AUTH_SECRET}:{password}".encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    password = password.strip()

    if not username or not password:
        return False, "Please enter both username and password."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = get_users_collection()
    if users.find_one({"username": username}):
        return False, "This username already exists."

    users.insert_one(
        {
            "username": username,
            "email": f"{username}@risk-shield.local",
            "password_hash": hash_password(password),
        }
    )
    return True, "Account created successfully."


def authenticate_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    password = password.strip()

    if not username or not password:
        return False, "Please enter both username and password."

    users = get_users_collection()
    user = users.find_one({"username": username})
    if not user:
        return False, "Invalid username or password."

    expected_hash = str(user.get("password_hash", ""))
    if not hmac.compare_digest(expected_hash, hash_password(password)):
        return False, "Invalid username or password."

    return True, username
