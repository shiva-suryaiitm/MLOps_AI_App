import os
import datetime
from flask import Flask, render_template, request, jsonify, Response
from pymongo import MongoClient
from dotenv import load_dotenv
from gridfs import GridFS
import time
import logging

# Ensure the logs directory exists
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

# Set up logging
log_file = os.path.join(log_dir, "web_application.log")
logging.basicConfig(
    level=logging.WARN,                      # Set the minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    filename=log_file,              # Log file name
    filemode='a',                            # Append mode ('w' would overwrite)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'             # Date format
)
logging.Formatter.converter = time.localtime
logger = logging.getLogger(__name__)

# Load environment variables
# load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

app = Flask(__name__)

# MongoDB connection
def get_mongodb_client():
    """Connect to MongoDB and return the client.
    Input: None
    Output: MongoDB client object or None on error"""
    try:
        mongo_host = os.getenv("DB_HOST", None)
        mongo_port = int(os.getenv("DB_PORT", None))
        
        connection_string = f"mongodb://{mongo_host}:{mongo_port}/"
        
        client = MongoClient(connection_string)
        client.admin.command('ping')
        logger.info(f"Connected to MongoDB at {mongo_host}:{mongo_port}")
        
        return client
    except Exception as e:
        logger.warning(f'Error connecting to mongodb: {e}')
        
        
# Get today's date for filtering
def get_today_date():
    """Get the current time and 24 hours ago for filtering.
    Input: None
    Output: Tuple of (current_time, 24_hours_ago)"""
    now = datetime.datetime.now()
    twenty_four_hours_ago = now - datetime.timedelta(hours=24)
    return now, twenty_four_hours_ago

@app.route('/')
def index():
    """Render the main index page.
    Input: None
    Output: Rendered HTML template"""
    return render_template('index.html')

@app.route('/api/image/<file_id>/<size>')
def get_image_with_size(file_id, size):
    """Retrieve and optionally resize an image from GridFS.
    Input: file_id (str), size (str, e.g., 'original' or 'WxH')
    Output: Flask Response with image data or error message"""
    try:
        from bson.objectid import ObjectId
        from PIL import Image
        import io
        
        client = get_mongodb_client()
        db_name = os.getenv("DB_NAME", None)
        
        db = client[db_name]
        fs = GridFS(db)
        file_id = ObjectId(file_id)
        grid_out = fs.get(file_id)
        data = grid_out.read()

        if size != 'original':
            img = Image.open(io.BytesIO(data))
            original_format = img.format  # Preserve original format

            # Parse target dimensions
            target_width, target_height = map(int, size.split('x'))

            # Calculate aspect ratio-preserving dimensions
            original_width, original_height = img.size
            ratio = min(target_width/original_width, target_height/original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)

            # Resize with high-quality filter
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Create new image with target size and neutral background
            background = Image.new(
                'RGB' if img.mode == 'RGB' else 'RGBA', 
                (target_width, target_height), 
                (255, 255, 255)  # Light gray background
            )
            
            # Calculate position to center the image
            position = (
                (target_width - new_width) // 2,
                (target_height - new_height) // 2
            )
            
            # Paste resized image onto background
            background.paste(img, position)
            img = background

            # Save to bytes with quality settings
            output = io.BytesIO()
            img.save(
                output, 
                format=original_format if original_format else 'JPEG',
                quality=85,  # Adjust quality (85 is good balance)
                optimize=True
            )
            data = output.getvalue()

        content_type = grid_out.content_type if hasattr(grid_out, 'content_type') else 'image/jpeg'
        return Response(data, mimetype=content_type)
    
    except Exception as e:
        logger.error(f"Error retrieving image {file_id}: {str(e)}")
        return jsonify({"error": str(e)}), 404

@app.route('/api/news')
def get_news():
    """Retrieve news articles published in the last 24 hours.
    Input: None
    Output: JSON response with news items or error message"""
    try:
        logger.info('getting news')
        client = get_mongodb_client()
        db_name = os.getenv("DB_NAME", None)
        collection_name = 'Metadata Collection'
        
        db = client[db_name]
        fs = GridFS(db)
        
        now, twenty_four_hours_ago = get_today_date()
        query = {
            "publication_timestamp": {"$gte": twenty_four_hours_ago, "$lt": now},
            "fs_files_image_id": {"$exists": True},
            "summary": {"$exists": True}
            }
        
        data_all =  [t for t in db['Metadata Collection'].find(query)]
        # to_del = db.fs.chunks.find()
        # for v in to_del:
        #     logger.error(f"MINEEE {v['_id']}")
        
        
        logger.info(f'{len(data_all)}')
        # Get news items
        news_items = []
        for item in data_all[::-1]:
            # Convert ObjectId to string for JSON serialization
            if '_id' in item:
                item['_id'] = str(item['_id'])
                
            t = get_image_with_size(file_id=item['fs_files_image_id'], size='original')
            logger.info(f'{ t }')
            # Extract only the fields we need
            filtered_item = {
                'title': item.get('title', ''),
                'publication_timestamp': item.get('publication_timestamp', ''),
                'url': item.get('url', ''),
                'image_url': item.get('image_url', ''),
                'image_id': str(item.get('fs_files_image_id', '')),
                'image_html_url': f"/api/image/{str(item.get('fs_files_image_id', ''))}",
                'tags': item.get('tags', []),
                'summary': item.get('summary', ''),
            }
            if filtered_item['summary'] != '' and filtered_item['summary'][-1] != '.':
                filtered_item['summary'] = filtered_item['summary'] + '.'
            
            news_items.append(filtered_item)
        
        return jsonify(news_items)
    
    except Exception as e:
        logger.error(f"Error fetching news: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if 'client' in locals():
            client.close()

@app.route('/api/health')
def health_check():
    """Health check endpoint.
    Input: None
    Output: JSON response with status"""
    return jsonify({"status": "healthy"})

def main():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    if os.getenv('DEBUG_MODE', 'False').lower() == 'true':
        logging.getLogger().setLevel(logging.INFO)
    
    # Make sure we can connect to MongoDB
    try:
        client = get_mongodb_client()
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
    

if __name__ == '__main__':
    main()