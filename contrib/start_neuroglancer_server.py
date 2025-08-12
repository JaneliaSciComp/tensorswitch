import http.server
import socketserver
import os

PORT = 8866

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    # Change to the directory where your data is located
    # For example, if your data is in 'test_for_AhrensLab/output_zarr3/'
    # you might want to change to that directory or a parent directory.
    # For now, I'll assume you want to serve from the project root.
    # You can modify this path if your data is elsewhere.
    os.chdir('test_for_AhrensLab/output_zarr3/') # Uncomment and modify if needed

    with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
        print(f"Serving Neuroglancer data at http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server.")
        httpd.serve_forever()
