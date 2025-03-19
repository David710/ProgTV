from flask import Flask, jsonify, render_template
import progtv
from datetime import datetime
from pathlib import Path
                 
app = Flask(__name__)                                        
@app.route('/')
def index():                                                                   
    tv_program = progtv.TVProgram()
    current_directory = Path.cwd()
    file_name_rated = f"{current_directory}/{tv_program.download_folder}/progtv_rated_{datetime.now().today().strftime('%Y-%m-%d')}.pkl"
    rated_progs = tv_program.read_programs(file_name_rated)
    prime_programs = tv_program.get_prime_programs(rated_progs)
    prime_programs_filtered = prime_programs[['name', 'start',  'icon', 'rating', 'cat', 'desc', 'note_pred','duration', 'channel_name', 'channel_icon']]
    return render_template('index.html', programs=prime_programs_filtered.to_dict(orient='records'))

@app.route('/api/programs')
def get_programs():
    tv_program = progtv.TVProgram()
    current_directory = Path.cwd()
    file_name_rated = f"{current_directory}/{tv_program.download_folder}/progtv_rated_{datetime.now().today().strftime('%Y-%m-%d')}.pkl"
    rated_progs = tv_program.read_programs(file_name_rated)
    prime_programs = tv_program.get_prime_programs(rated_progs)
    prime_programs_filtered = prime_programs[['name', 'start',  'icon', 'rating', 'cat', 'desc', 'note_pred','duration', 'channel_name', 'channel_icon']]
    return jsonify(prime_programs_filtered.to_dict(orient='records'))

@app.route('/api/suggestions')
def get_suggestions():
    best_programs_filtered = prepare_suggestions()
    return jsonify(best_programs_filtered.to_dict(orient='records'))

def prepare_suggestions():
    number_of_suggestions = 5
    channels = ['TF1', 'France 2', 'France 3']
    tv_program = progtv.TVProgram()
    current_directory = Path.cwd()
    file_name_rated = f"{current_directory}/{tv_program.download_folder}/progtv_rated_{datetime.now().today().strftime('%Y-%m-%d')}.pkl"
    rated_progs = tv_program.read_programs(file_name_rated)
    best_programs = tv_program.get_best_programs(rated_progs, n=number_of_suggestions, whitelist=channels)
    best_programs_filtered = best_programs[['name', 'start',  'icon', 'rating', 'cat', 'desc', 'note_pred','duration', 'channel_name', 'channel_icon']]
    print(best_programs_filtered)
    return best_programs_filtered

@app.route('/api/ai_comments')
def get_ai_comments():
    best_programs_filtered = prepare_suggestions()
    tv_program = progtv.TVProgram()
    comments = tv_program.add_ollama_comment_to_dataset(best_programs_filtered)
    return jsonify(comments.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)