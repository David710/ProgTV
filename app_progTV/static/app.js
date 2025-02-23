function formatAsPercentage(decimal) {
    return (decimal * 100).toFixed(0) + '%';
}

function formatDuration(minutes) {
    if (minutes >= 60) {
        const hours = Math.floor(minutes / 60);
        const remainingMinutes = minutes % 60;
        return `${hours}h ${remainingMinutes}min`;
    } else {
        return `${minutes}min`;
    }
}

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
                programGroup.classList.add("row", "mb-3");
                programsDiv.appendChild(programGroup);
            }
            const programElement = document.createElement('div');
            programElement.classList.add("card", "p-2", "col-md", "me-3");

            // Subtract one hour from program.start
            const startTime = new Date(program.start);
            startTime.setHours(startTime.getHours() - 1);
            const formattedStartTime = startTime.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });

            // Format note_pred as a percentage
            const formattedNotePred = formatAsPercentage(program.note_pred);

            // Format duration
            const formattedDuration = formatDuration(program.duration);

            programElement.innerHTML = `
                <img src="${program.icon}" alt="Program Image" class="card-img-top">
                <div class="card-body">
                    <div class="d-flex justify-content-start align-items-center flex-row mb-3">
                        <img src="${program.channel_icon}" alt="Channel Icon" class="channel-icon p-2">
                        <p class="card-title p-2"><span class="badge text-bg-light">${formattedStartTime}</span></p>
                        <h5 class="card-title p-2">${program.name}</h5>
                    </div>
                    <p class="card-text">${program.desc}</p>
                    <div class="d-flex justify-content-start align-items-center flex-row mb-3">
                        <p class="card-text p-2 info-add"><span class="badge text-bg-primary">${program.rating}</span></p>
                        <p class="card-text p-2 info-add"><span class="badge text-bg-secondary">${program.cat}</span></p>
                        <p class="card-text p-2 info-add"><span class="badge text-bg-danger">${formattedNotePred}</span></p>
                        <p class="card-text p-2 info-add"><span class="badge text-bg-light">${formattedDuration}</span></p>
                    </div>
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