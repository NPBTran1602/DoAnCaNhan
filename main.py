from environment_selector import select_environment
from app import initialize_app

def main():
    select_environment(initialize_app)

if __name__ == "__main__":
    main()