import logging
# logging
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(message)s', 
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[
        logging.FileHandler('./rough_sketch/log/app.log'),
        logging.StreamHandler()
    ]
)
# logger
logger = logging.getLogger(__name__)