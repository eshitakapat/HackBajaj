import sys
from app.config import settings
from DataBase.database import init_pool, get_connection_params

def test_connection():
    try:
        # Test getting connection parameters
        print("🔍 Testing database connection parameters...")
        params = get_connection_params()
        print("✅ Connection parameters:")
        for key in ['host', 'port', 'dbname', 'user']:
            print(f"   {key}: {params[key]}")
        
        # Test creating a connection pool
        print("\n🔌 Testing database connection...")
        pool = init_pool()
        assert pool is not None, "Failed to create connection pool"
        print("✅ Successfully connected to the database!")
        
        # Test a simple query
        with pool.getconn() as conn:
            with conn.cursor() as cur:
                # Test database version
                cur.execute("SELECT version()")
                version = cur.fetchone()
                assert version is not None, "Failed to fetch database version"
                print(f"\n📊 Database version: {version['version']}")
                
                # Test basic query
                cur.execute("SELECT 1 + 1 as result")
                result = cur.fetchone()
                assert result is not None and result['result'] == 2, "Basic query test failed"
                print(f"✅ Basic query test passed: 1 + 1 = {result['result']}")
                
                # Test if pgvector extension is available
                try:
                    cur.execute("SELECT 1")
                    test_result = cur.fetchone()
                    assert test_result is not None and test_result[0] == 1, "Basic connectivity test failed"
                    print("✅ Basic database connectivity test passed")
                    
                except Exception as e:
                    print(f"⚠️ Basic database connectivity test failed: {str(e)}")
                    
    except Exception as e:
        print(f"❌ Error testing database connection: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_connection()
