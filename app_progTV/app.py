from flask import Flask
import progtv
app = Flask(__name__)
from datetime import datetime
from pathlib import Path

@app.route('/')
def index():
    tv_program = progtv.TVProgram()
    current_directory = Path.cwd()
    file_name_rated = f"{current_directory}//{tv_program.download_folder}/progtv_rated_{datetime.now().today().strftime('%Y-%m-%d')}.pkl"
    rated_progs = tv_program.read_programs(file_name_rated)
    prime_programs = tv_program.get_prime_programs(rated_progs)
    print(prime_programs)
    return prime_programs.to_html()

if __name__ == '__main__':
    app.run(debug=True)