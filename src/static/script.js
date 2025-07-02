const queryForm = document.getElementById('query-form');
const queryInput = document.getElementById('query-input');
const responseContainer = document.getElementById('response-container');

queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value;
    if (!query) return;

    responseContainer.textContent = 'Thinking...';

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        responseContainer.textContent = data.response;
    } catch (error) {
        console.error('Error fetching response:', error);
        responseContainer.textContent = 'An error occurred. Please check the console for details.';
    }
});
