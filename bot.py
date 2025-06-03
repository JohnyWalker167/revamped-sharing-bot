import uuid
import asyncio
import requests
import base64
import re
import sys
from datetime import datetime, timezone, timedelta
from pyrogram import Client, enums, filters
from pyrogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import uvicorn
from config import *

# MongoDB setup
mongo = MongoClient(MONGO_URI)
db = mongo["sharing_bot"]
files_col = db["files"]
tokens_col = db["tokens"]
auth_users_col = db["auth_users"]
allowed_channels_col = db["allowed_channels"]

TOKEN_VALIDITY_SECONDS = 24 * 60 * 60  # 24 hours

# Pyrogram bot initialization
bot = Client(
    "bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    workers=1000,
    parse_mode=enums.ParseMode.HTML
)

# FastAPI app
api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=[f"{MY_DOMAIN}"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for file lists
channel_files_cache = {}

# --- Utility Functions ---

async def get_allowed_channels():
    return [doc["channel_id"] for doc in allowed_channels_col.find({}, {"_id": 0, "channel_id": 1})]

def generate_telegram_link(bot_username, channel_id, message_id):
    """Generate a base64-encoded Telegram deep link for a file."""
    raw = f"{channel_id}_{message_id}".encode()
    b64 = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"https://telegram.dog/{bot_username}?start=file_{b64}"

def generate_token(user_id):
    """Generate a new access token for a user."""
    token_id = str(uuid.uuid4())
    expiry = datetime.now(timezone.utc) + timedelta(seconds=TOKEN_VALIDITY_SECONDS)
    tokens_col.insert_one({
        "token_id": token_id,
        "user_id": user_id,
        "expiry": expiry,
        "created_at": datetime.now(timezone.utc)
    })
    return token_id

def is_token_valid(token_id, user_id):
    """Check if a token is valid for a user."""
    token = tokens_col.find_one({"token_id": token_id, "user_id": user_id})
    if not token:
        return False
    expiry = token["expiry"]
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    if expiry < datetime.now(timezone.utc):
        tokens_col.delete_one({"_id": token["_id"]})
        return False
    return True

def get_token_link(token_id, bot_username):
    """Generate a Telegram deep link for a token."""
    return f"https://telegram.dog/{bot_username}?start=token_{token_id}"

def shorten_url(long_url):
    """Shorten a URL using the configured shortener."""
    try:
        resp = requests.get(f"https://{SHORTERNER_URL}/api?api={URLSHORTX_API_TOKEN}&url={long_url}", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success" and data.get("shortenedUrl"):
                return data["shortenedUrl"]
        return long_url
    except Exception:
        return long_url

def authorize_user(user_id):
    """Authorize a user for 24 hours."""
    expiry = datetime.now(timezone.utc) + timedelta(seconds=TOKEN_VALIDITY_SECONDS)
    auth_users_col.update_one(
        {"user_id": user_id},
        {"$set": {"expiry": expiry}},
        upsert=True
    )

def is_user_authorized(user_id):
    """Check if a user is authorized."""
    doc = auth_users_col.find_one({"user_id": user_id})
    if not doc:
        return False
    expiry = doc["expiry"]
    if isinstance(expiry, str):
        try:
            expiry = datetime.fromisoformat(expiry)
        except Exception:
            return False
    if isinstance(expiry, datetime) and expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    if expiry < datetime.now(timezone.utc):
        return False
    return True

def upsert_file_info(file_info):
    """Insert or update file info, avoiding duplicates."""
    files_col.update_one(
        {"channel_id": file_info["channel_id"], "message_id": file_info["message_id"]},
        {"$set": file_info},
        upsert=True
    )

def human_readable_size(size):
    for unit in ['B','KB','MB','GB','TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def invalidate_channel_cache(channel_id):
    keys_to_delete = [k for k in channel_files_cache if k.startswith(f"{channel_id}:")]
    for k in keys_to_delete:
        del channel_files_cache[k]

# --- Bot Command Handlers ---

@bot.on_message(filters.command("start") & filters.private)
async def start_handler(client, message):
    user_id = message.from_user.id
    bot_username = (await bot.get_me()).username

    # Token-based authorization
    if len(message.command) == 2 and message.command[1].startswith("token_"):
        token_id = message.command[1][6:]
        if is_token_valid(token_id, user_id):
            authorize_user(user_id)
            await message.reply_text("✅ You are now authorized to access files for 24 hours.")
        else:
            await message.reply_text("❌ Invalid or expired token. Please get a new link.")
        return

    # File access via deep link
    if len(message.command) == 2 and message.command[1].startswith("file_"):
        if not is_user_authorized(user_id):
            now = datetime.now(timezone.utc)
            token_doc = tokens_col.find_one({
                "user_id": user_id,
                "expiry": {"$gt": now}
            })
            if token_doc:
                token_id = token_doc["token_id"]
            else:
                token_id = generate_token(user_id)
            short_link = shorten_url(get_token_link(token_id, bot_username))
            await message.reply_text(
                "❌ You are not authorized or your access expired.\n"
                "Please use this link to get access for 24 hours:",
                reply_markup=InlineKeyboardMarkup(
                    [[InlineKeyboardButton("Get Access Link", url=short_link)]]
                )
            )
            return
        try:
            b64 = message.command[1][5:]
            padding = '=' * (-len(b64) % 4)
            decoded = base64.urlsafe_b64decode(b64 + padding).decode()
            channel_id_str, msg_id_str = decoded.split("_")
            channel_id = int(channel_id_str)
            msg_id = int(msg_id_str)
        except Exception:
            await message.reply_text("Invalid file link.")
            return
        file_doc = files_col.find_one({"channel_id": channel_id, "message_id": msg_id})
        if not file_doc:
            await message.reply_text("File not found.")
            return
        try:
            await client.copy_message(
                chat_id=message.chat.id,
                from_chat_id=file_doc["channel_id"],
                message_id=file_doc["message_id"]
            )
        except Exception as e:
            await message.reply_text(f"Failed to send file: {e}")
        return

    # Default greeting
    await message.reply_text(
        "👋 Hello! Welcome to the bot. Use a valid access link to get file access."
    )

@bot.on_message(filters.document | filters.video | filters.audio | filters.photo)
async def channel_file_handler(client, message):
    """Save files sent in allowed channels only, using caption as file name if available."""
    try:
        allowed_channels = await get_allowed_channels()
        if message.chat.id not in allowed_channels:
            return  # Ignore files from non-allowed channels and private chats

        # Prefer caption as file name, fallback to actual file name
        caption_name = message.caption.strip() if message.caption else None

        file_info = {
            "channel_id": message.chat.id,
            "message_id": message.id,
            "file_name": None,
            "file_size": None,
            "file_format": None,
            "date": datetime.now(timezone.utc)
        }
        if message.document:
            file_info["file_name"] = caption_name or message.document.file_name
            file_info["file_size"] = message.document.file_size
            file_info["file_format"] = message.document.mime_type
        elif message.video:
            file_info["file_name"] = caption_name or (message.video.file_name or "video.mp4")
            file_info["file_size"] = message.video.file_size
            file_info["file_format"] = message.video.mime_type
        elif message.audio:
            file_info["file_name"] = caption_name or (message.audio.file_name or "audio.mp3")
            file_info["file_size"] = message.audio.file_size
            file_info["file_format"] = message.audio.mime_type
        elif message.photo:
            file_info["file_name"] = caption_name or "photo.jpg"
            file_info["file_size"] = getattr(message.photo, "file_size", None)
            file_info["file_format"] = "image/jpeg"
        upsert_file_info(file_info)
    except Exception as e:
        await message.reply_text(f"❌ Error saving file: {e}")

# --- Index channel files using file link and range ---

@bot.on_message(filters.command("index") & filters.user(OWNER_ID))
async def index_channel_files(client, message: Message):
    """
    Interactive: Asks for start and end file links, then indexes files in that range.
    Only works for allowed channels (from DB).
    """
    await message.reply_text("Please send the **start file link** (Telegram message link):")
    start_msg = await client.listen(message.chat.id, timeout=120)
    start_link = start_msg.text.strip()

    await message.reply_text("Now send the **end file link** (Telegram message link):")
    end_msg = await client.listen(message.chat.id, timeout=120)
    end_link = end_msg.text.strip()

    def extract_channel_and_msg_id(link):
        # t.me/c/<channel_id>/<msg_id>
        match = re.search(r"t\.me/c/(-?\d+)/(\d+)", link)
        if match:
            channel_id = int("-100" + match.group(1)) if not match.group(1).startswith("-100") else int(match.group(1))
            msg_id = int(match.group(2))
            return channel_id, msg_id
        # t.me/<username>/<msg_id>
        match = re.search(r"t\.me/([\w\d_]+)/(\d+)", link)
        if match:
            username = match.group(1)
            msg_id = int(match.group(2))
            return username, msg_id
        raise ValueError("Invalid Telegram message link format.")

    try:
        start_id, start_msg_id = extract_channel_and_msg_id(start_link)
        end_id, end_msg_id = extract_channel_and_msg_id(end_link)

        # If username, resolve to channel_id
        if isinstance(start_id, str):
            chat = await client.get_chat(start_id)
            start_id = chat.id
        if isinstance(end_id, str):
            chat = await client.get_chat(end_id)
            end_id = chat.id

        if start_id != end_id:
            await message.reply_text("Start and end links must be from the same channel.")
            return

        channel_id = start_id
        allowed_channels = await get_allowed_channels()
        if channel_id not in allowed_channels:
            await message.reply_text("❌ This channel is not allowed for indexing.")
            return

        if start_msg_id > end_msg_id:
            start_msg_id, end_msg_id = end_msg_id, start_msg_id

    except Exception as e:
        await message.reply_text(f"Invalid link: {e}")
        return

    await message.reply_text(f"Indexing files from {start_msg_id} to {end_msg_id} in channel {channel_id}...")

    batch_size = 199
    total_saved = 0
    for batch_start in range(start_msg_id, end_msg_id + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, end_msg_id)
        ids = list(range(batch_start, batch_end + 1))
        try:
            await asyncio.sleep(3)  # Avoid flood
            messages = await client.get_messages(channel_id, ids)
        except Exception as e:
            await message.reply_text(f"Failed to get messages {batch_start}-{batch_end}: {e}")
            continue
        for msg in messages:
            if not msg:
                continue
            if msg.document or msg.video or msg.audio or msg.photo:
                caption_name = msg.caption.strip() if msg.caption else None
                file_info = {
                    "channel_id": channel_id,
                    "message_id": msg.id,
                    "file_name": None,
                    "file_size": None,
                    "file_format": None,
                    "date": msg.date.replace(tzinfo=timezone.utc) if msg.date else datetime.now(timezone.utc)
                }
                if msg.document:
                    file_info["file_name"] = caption_name or msg.document.file_name
                    file_info["file_size"] = msg.document.file_size
                    file_info["file_format"] = msg.document.mime_type
                elif msg.video:
                    file_info["file_name"] = caption_name or msg.video.file_name
                    file_info["file_size"] = msg.video.file_size
                    file_info["file_format"] = msg.video.mime_type
                elif msg.audio:
                    file_info["file_name"] = caption_name or msg.audio.file_name
                    file_info["file_size"] = msg.audio.file_size
                    file_info["file_format"] = msg.audio.mime_type
                elif msg.photo:
                    file_info["file_name"] = caption_name or "photo.jpg"
                    file_info["file_size"] = getattr(msg.photo, "file_size", None)
                    file_info["file_format"] = "image/jpeg"
                upsert_file_info(file_info)
                total_saved += 1
        invalidate_channel_cache(channel_id)
        await asyncio.sleep(24)  # avoid flood

    await message.reply_text(f"✅ Indexed {total_saved} files from channel {channel_id}.")

@bot.on_message(filters.command("stats") & filters.private & filters.user(OWNER_ID))
async def stats_command(client, message: Message):
    """Show statistics (only for OWNER_ID)."""
    total_users = auth_users_col.count_documents({})
    total_files = files_col.count_documents({})
    pipeline = [
        {"$group": {"_id": None, "total": {"$sum": "$file_size"}}}
    ]
    result = list(files_col.aggregate(pipeline))
    total_storage = result[0]["total"] if result else 0

    stats = db.command("dbstats")
    db_storage = stats.get("storageSize", 0)

    await message.reply_text(
        f"👤 Total users: <b>{total_users}</b>\n"
        f"📁 Total files: <b>{total_files}</b>\n"
        f"💾 Files size: <b>{human_readable_size(total_storage)}</b>\n"
        f"📊 Database storage used: <b>{db_storage / (1024 * 1024):.2f} MB</b>",
    )

# --- Add Channel Command ---
@bot.on_message(filters.command("addchannel") & filters.user(OWNER_ID))
async def add_channel_handler(client, message: Message):
    """
    Usage: /addchannel <channel_id> or /addchannel <telegram_message_link>
    Adds a channel to the allowed channels list in the database.
    """
    if len(message.command) != 2:
        await message.reply_text("Usage: /addchannel <channel_id> or /addchannel <telegram_message_link>")
        return
    arg = message.command[1]
    try:
        if arg.startswith("https://t.me/"):
            # Try to extract channel_id from link
            match = re.search(r"t\.me/c/(-?\d+)", arg)
            if match:
                channel_id = int("-100" + match.group(1)) if not match.group(1).startswith("-100") else int(match.group(1))
            else:
                match = re.search(r"t\.me/([\w\d_]+)", arg)
                if match:
                    username = match.group(1)
                    chat = await client.get_chat(username)
                    channel_id = chat.id
                else:
                    await message.reply_text("Invalid Telegram message link.")
                    return
        else:
            channel_id = int(arg)
        allowed_channels_col.update_one(
            {"channel_id": channel_id},
            {"$set": {"channel_id": channel_id}},
            upsert=True
        )
        await message.reply_text(f"✅ Channel {channel_id} added to allowed channels.")
    except Exception as e:
        await message.reply_text(f"Error: {e}")

# --- Remove Channel Command ---
@bot.on_message(filters.command("removechannel") & filters.user(OWNER_ID))
async def remove_channel_handler(client, message: Message):         
    """
    Usage: /removechannel <channel_id> or /removechannel <telegram_message_link>
    Removes a channel from the allowed channels list in the database.
    """
    if len(message.command) != 2:
        await message.reply_text("Usage: /removechannel <channel_id> or /removechannel <telegram_message_link>")
        return
    arg = message.command[1]
    try:
        if arg.startswith("https://t.me/"):
            # Try to extract channel_id from link
            match = re.search(r"t\.me/c/(-?\d+)", arg)
            if match:
                channel_id = int("-100" + match.group(1)) if not match.group(1).startswith("-100") else int(match.group(1))
            else:
                match = re.search(r"t\.me/([\w\d_]+)", arg)
                if match:
                    username = match.group(1)
                    chat = await client.get_chat(username)
                    channel_id = chat.id
                else:
                    await message.reply_text("Invalid Telegram message link.")
                    return
        else:
            channel_id = int(arg)
        result = allowed_channels_col.delete_one({"channel_id": channel_id})
        invalidate_channel_cache(channel_id)
        if result.deleted_count:
            await message.reply_text(f"✅ Channel {channel_id} removed from allowed channels.")
        else:
            await message.reply_text("❌ Channel not found in allowed channels.")
    except Exception as e:
        await message.reply_text(f"Error: {e}")

# --- Delete File Command ---
@bot.on_message(filters.command("delete") & filters.user(OWNER_ID))
async def delete_file_handler(client, message: Message):
    """
    Usage:
      /delete <telegram_message_link>
      or: /delete <channel_id> <message_id>
    Deletes a file from the database using a Telegram message link or IDs.
    """
    # If command has a Telegram message link
    if len(message.command) == 2 and message.command[1].startswith("https://t.me/"):
        link = message.command[1]
        try:
            # t.me/c/<id>/<msg_id>
            match = re.search(r"t\.me/c/(-?\d+)/(\d+)", link)
            if match:
                channel_id = int("-100" + match.group(1)) if not match.group(1).startswith("-100") else int(match.group(1))
                message_id = int(match.group(2))
            else:
                # t.me/<username>/<msg_id>
                match = re.search(r"t\.me/([\w\d_]+)/(\d+)", link)
                if match:
                    username = match.group(1)
                    message_id = int(match.group(2))
                    chat = await client.get_chat(username)
                    channel_id = chat.id
                else:
                    await message.reply_text("Invalid Telegram message link.")
                    return
        except Exception:
            await message.reply_text("Invalid Telegram message link.")
            return
    # Or if command has channel_id and message_id
    elif len(message.command) == 3:
        try:
            channel_id = int(message.command[1])
            message_id = int(message.command[2])
        except Exception:
            await message.reply_text("Usage: /delete <telegram_message_link> or /delete <channel_id> <message_id>")
            return
    else:
        await message.reply_text("Usage: /delete <telegram_message_link> or /delete <channel_id> <message_id>")
        return

    try:
        result = files_col.delete_one({"channel_id": channel_id, "message_id": message_id})
        invalidate_channel_cache(channel_id)
        if result.deleted_count:
            await message.reply_text(f"✅ File ({channel_id}, {message_id}) deleted from database.")
        else:
            await message.reply_text("❌ File not found in database.")
    except Exception as e:
        await message.reply_text(f"Error: {e}")

@bot.on_message(filters.command('restart') & filters.private & filters.user(OWNER_ID))
async def restart(client, message):
    os.system("python3 update.py")  
    os.execl(sys.executable, sys.executable, "bot.py")
                
# --- FastAPI Endpoints ---

@api.get("/")
async def root():
    """Greet users on root route."""
    return JSONResponse({"message": "👋 Hello! Welcome to the Sharing Bot"})

@api.get("/api/channels")
async def api_channels():
    """List all channels (JSON)."""
    channels = list(files_col.aggregate([
        {"$group": {"_id": "$channel_id"}}
    ]))
    channel_infos = []
    for ch in channels:
        channel_id = ch["_id"]
        try:
            chat = await bot.get_chat(channel_id)
            channel_name = chat.title or str(channel_id)
        except Exception:
            channel_name = str(channel_id)
        channel_infos.append({"id": channel_id, "name": channel_name})
    return JSONResponse({"channels": channel_infos})

@api.get("/api/channel/{channel_id}/files")
async def api_channel_files(
    channel_id: int,
    q: str = "",
    offset: int = 0,
    limit: int = 10
):
    """List files for a channel (JSON)."""
    bot_username = (await bot.get_me()).username
    query = {"channel_id": channel_id}
    if q:
        regex = ".*".join(map(lambda s: s, q.strip().split()))
        query["file_name"] = {"$regex": regex, "$options": "i"}

    cache_key = f"{channel_id}:{q}:{offset}:{limit}"
    if cache_key not in channel_files_cache:
        files = list(files_col.find(query, {"_id": 0}).sort("message_id", -1))
        for file in files:
            file["telegram_link"] = generate_telegram_link(bot_username, file["channel_id"], file["message_id"])
            if isinstance(file.get("date"), str):
                try:
                    file["date"] = datetime.fromisoformat(file["date"])
                except Exception:
                    file["date"] = None
        channel_files_cache[cache_key] = files
    else:
        files = channel_files_cache[cache_key]

    paginated_files = files[offset:offset+limit]
    has_more = offset + limit < len(files)

    def serialize_file(file):
        return {
            "file_name": file.get("file_name"),
            "file_size": file.get("file_size"),
            "file_format": file.get("file_format"),
            "date": file.get("date").strftime('%Y-%m-%d %H:%M:%S') if file.get("date") else "",
            "telegram_link": file.get("telegram_link")
        }

    return JSONResponse({
        "files": [serialize_file(f) for f in paginated_files],
        "has_more": has_more
    })

# --- Main Entrypoint ---

async def main():
    await bot.start()
    bot.loop.create_task(start_fastapi())

async def start_fastapi():
    config = uvicorn.Config(api, host="0.0.0.0", port=8000, loop="asyncio")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

if __name__ == "__main__":
    try:
        bot.loop.run_until_complete(main())
        bot.loop.run_forever()
    except KeyboardInterrupt:
        bot.stop()
        tasks = asyncio.all_tasks(loop=bot.loop)
        for task in tasks:
            task.cancel()
        bot.loop.stop()