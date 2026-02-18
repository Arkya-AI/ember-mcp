import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("init", "status"):
        from ember.cli import main
        main()
    else:
        from ember.server import mcp
        mcp.run()
