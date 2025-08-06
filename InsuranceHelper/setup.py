from setuptools import setup, find_packages

setup(
    name="insurance_helper",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'pydantic',
        'python-jose',
        'passlib',
        'python-multipart',
        'email-validator',
        'python-dotenv',
    ],
)
