// Fetch the JSON data from the Flask endpoint
fetch('/api/programs')
    .then(response => response.json())
    .then(data => {
        // Process the JSON data
        console.log(data);
        const programsDiv = document.getElementById('programs');
        let iRow = 0;
        let programGroup;
        data.forEach(program => {
            if (iRow % 3 === 0) {
                programGroup = document.createElement('div');
                programGroup.classList.add("row");
                programsDiv.appendChild(programGroup);
            }
            const programElement = document.createElement('div');
            programElement.classList.add("card", "p-2", "col-md", "me-3");
            programElement.innerHTML = `
                <img src="${program.icon}" alt="Program Image" class="card-img-top">
                <div class="card-body">
                    <h5 class="card-title">${program.name}</h5>
                    <p class="card-text">${program.desc}</p>
                    <p>Start: ${program.start}</p>
                    <p>Rating: ${program.rating}</p>
                    <p>Category: ${program.cat}</p>
                    <p>Note: ${program.note_pred}</p>
                    <p>Duration: ${program.duration} minutes</p>
                    <p>Channel: ${program.channel_name}</p>
                </div>
            `;
            programGroup.appendChild(programElement);
            iRow++;
        });
    })
    .catch(error => console.error('Error fetching data:', error));

// Get the current date
const currentDate = new Date();
const options = { weekday: 'long', day: 'numeric', month: 'long' };
const formattedDate = currentDate.toLocaleDateString('fr-FR', options);

// Update the page title with the current day and month
const pageTitle = document.getElementById('page-title');
pageTitle.innerHTML = `Programmes du ${formattedDate}`;

{/* <h2>${program.name}</h2>
<p>Start: ${program.start}</p>
<p>Rating: ${program.rating}</p>
<p>Category: ${program.cat}</p>
<p>Description: ${program.desc}</p>
<p>Note: ${program.note_pred}</p>
<p>Duration: ${program.duration} minutes</p>
<p>Channel: ${program.channel_name}</p>
<img src="${program.icon}" alt="Channel Icon"> */}