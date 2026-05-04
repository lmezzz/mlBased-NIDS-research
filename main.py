from src.web.app import create_app


def main():
    app = create_app()
    print("[IDS] Starting Flask demo at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
