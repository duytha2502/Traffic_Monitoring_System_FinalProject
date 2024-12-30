import pymongo
import gridfs

def connect_to_mongo(db_name):
    """
    Kết nối tới MongoDB và trả về đối tượng database, collection và GridFS.

    Args:
        db_name (str): Tên database cần kết nối.

    Returns:
        db: Đối tượng database.
        fs: Đối tượng GridFS.
    """
    # Địa chỉ MongoDB Local
    mongo_uri = "mongodb://localhost:27017/"
    
    # Kết nối tới MongoDB
    client = pymongo.MongoClient(mongo_uri)
    
    # Truy cập vào database
    db = client[db_name]
    
    # Tạo GridFS cho database
    fs = gridfs.GridFS(db)
    
    return db, fs