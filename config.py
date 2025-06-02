
import os
from dotenv import load_dotenv

load_dotenv('config.env', override=True)

#TELEGRAM API
API_ID = int(os.getenv('API_ID'))
API_HASH = os.getenv('API_HASH')
BOT_TOKEN = os.getenv('BOT_TOKEN')
OWNER_ID = int(os.getenv('OWNER_ID'))

TOKEN_VALIDITY_SECONDS = 24 * 60 * 60  # 24 hours

MONGO_URI = os.getenv("MONGO_URI")

#SHORTERNER API
URLSHORTX_API_TOKEN = os.getenv('URLSHORTX_API_TOKEN')
SHORTERNER_URL = os.getenv('SHORTERNER_URL')