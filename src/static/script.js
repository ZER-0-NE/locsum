document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.getElementById('query-input');
    const submitButton = document.getElementById('submit-button');
    const responseContainer = document.getElementById('response-container');
    const indexStatusText = document.getElementById('index-status-text');
    const switchIndexButton = document.getElementById('switch-index-button');

    // Function to update index status
    const updateIndexStatus = async () => {
        try {
            const response = await fetch('/index-status');
            const data = await response.json();

            indexStatusText.textContent = `Index Status: Active (${data.blue_index.toUpperCase()})`;

            if (data.green_index_ready_for_swap) {
                switchIndexButton.classList.remove('hidden');
                indexStatusText.textContent += ` (New index ${data.green_index.toUpperCase()} ready!)`;
            } else {
                switchIndexButton.classList.add('hidden');
            }
        } catch (error) {
            console.error('Error fetching index status:', error);
            indexStatusText.textContent = 'Index Status: Error';
            switchIndexButton.classList.add('hidden');
        }
    };

    // Initial status update and then poll every 5 seconds
    updateIndexStatus();
    setInterval(updateIndexStatus, 5000);

    // Event listener for the switch index button
    switchIndexButton.addEventListener('click', async () => {
        switchIndexButton.disabled = true; // Disable button to prevent multiple clicks
        switchIndexButton.textContent = 'Switching...';
        try {
            const response = await fetch('/switch-index', {
                method: 'POST',
            });
            const data = await response.json();
            alert(data.message);
            updateIndexStatus(); // Update status after switch
        } catch (error) {
            console.error('Error switching index:', error);
            alert('Failed to switch index. Check console for details.');
        } finally {
            switchIndexButton.disabled = false;
            switchIndexButton.textContent = 'Switch to New Index';
        }
    });

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