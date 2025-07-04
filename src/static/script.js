document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.getElementById('query-input');
    const submitButton = document.getElementById('submit-button');
    const responseContainer = document.getElementById('response-container');

    submitButton.addEventListener('click', async () => {
        const query = queryInput.value.trim();
        if (!query) return;

        // Display user message
        const userMessageDiv = document.createElement('div');
        userMessageDiv.classList.add('message', 'user-message');
        userMessageDiv.textContent = query;
        responseContainer.appendChild(userMessageDiv);

        queryInput.value = ''; // Clear input
        responseContainer.scrollTop = responseContainer.scrollHeight; // Scroll to bottom

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Display AI response
            const aiMessageDiv = document.createElement('div');
            aiMessageDiv.classList.add('message', 'ai-message');
            aiMessageDiv.textContent = data.response; // Assuming the backend sends a 'response' field
            responseContainer.appendChild(aiMessageDiv);

        } catch (error) {
            console.error('Error:', error);
            const errorMessageDiv = document.createElement('div');
            errorMessageDiv.classList.add('message', 'ai-message');
            errorMessageDiv.style.color = 'red';
            errorMessageDiv.textContent = `Error: ${error.message}. Please try again.`;
            responseContainer.appendChild(errorMessageDiv);
        }
        responseContainer.scrollTop = responseContainer.scrollHeight; // Scroll to bottom
    });

    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            submitButton.click();
        }
    });
});