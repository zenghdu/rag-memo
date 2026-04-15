from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

# MySQL 连接配置
# 支持 mysql+pymysql://user:password@host:port/db
engine = create_engine(
    settings.mysql_uri,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    echo=False # 如果需要查看原生 SQL, 改为 True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """FastAPI Dependency for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """初始化数据库表"""
    from app.core.models import Base
    Base.metadata.create_all(bind=engine)
