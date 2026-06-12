/**
 * FallSync Frontend Application
 * Real-time fall detection dashboard
 */

// Configuration
const API_BASE_URL = window.location.origin;
const RECONNECT_INTERVAL = 5000; // 5 seconds

// DOM Elements
const statusIndicator = document.getElementById('status-indicator');
const statusText = document.getElementById('status-text');
const currentStatusDiv = document.getElementById('current-status');
const fallProbabilityDiv = document.getElementById('fall-probability');
const confidenceDiv = document.getElementById('confidence');
const peakAccelerationDiv = document.getElementById('peak-acceleration');
const dataInput = document.getElementById('data-input');
const predictBtn = document.getElementById('predict-btn');
const demoBtn = document.getElementById('demo-btn');
const clearBtn = document.getElementById('clear-btn');
const resultsContainer = document.getElementById('results-container');
const logContainer = document.getElementById('log-container');
const accelerationCanvas = document.getElementById('acceleration-chart');

// State
let modelReady = false;
let chart = null;

/**
 * Initialize application
 */
function init() {
    setupEventListeners();
    checkModelStatus();
    generateDemoData();
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    predictBtn.addEventListener('click', handlePredict);
    demoBtn.addEventListener('click', handleLoadDemo);
    clearBtn.addEventListener('click', handleClear);
    dataInput.addEventListener('paste', handlePaste);
}

/**
 * Check model status
 */
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        updateStatus(data.model_ready);
        addLog(`Model status: ${data.model_ready ? 'Ready' : 'Not initialized'}`, 
               data.model_ready ? 'success' : 'warning');
    } catch (error) {
        updateStatus(false);
        addLog(`Connection error: ${error.message}`, 'error');
        setTimeout(checkModelStatus, RECONNECT_INTERVAL);
    }
}

/**
 * Update status indicator
 */
function updateStatus(ready) {
    modelReady = ready;
    statusIndicator.className = `status-dot ${ready ? 'connected' : 'error'}`;
    statusText.textContent = ready ? 'Connected' : 'Disconnected';
    predictBtn.disabled = !ready;
}

/**
 * Handle predict button click
 */
async function handlePredict() {
    const data = dataInput.value.trim();
    
    if (!data) {
        addLog('Please enter accelerometer data', 'warning');
        return;
    }
    
    try {
        // Parse JSON
        let accelerometerData;
        try {
            accelerometerData = JSON.parse(data);
        } catch (e) {
            addLog(`Invalid JSON: ${e.message}`, 'error');
            return;
        }
        
        // Validate data
        if (!Array.isArray(accelerometerData)) {
            addLog('Data must be an array of [x, y, z] values', 'error');
            return;
        }
        
        if (accelerometerData.length < 60) {
            addLog(`Insufficient data: ${accelerometerData.length}/60 samples`, 'warning');
            return;
        }
        
        // Send prediction request
        predictBtn.disabled = true;
        predictBtn.textContent = '⏳ Processing...';
        addLog('Sending prediction request...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                accelerometer_data: accelerometerData
            })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            addLog(`Error: ${result.error || 'Unknown error'}`, 'error');
            return;
        }
        
        // Display results
        displayResults(result);
        displayChart(accelerometerData);
        
        // Add log entry
        const status = result.is_fall ? '🚨 FALL DETECTED' : '✅ No Fall';
        addLog(`Prediction complete: ${status}`, result.is_fall ? 'error' : 'success');
        
    } catch (error) {
        addLog(`Request error: ${error.message}`, 'error');
    } finally {
        predictBtn.disabled = !modelReady;
        predictBtn.textContent = '📊 Predict Fall';
    }
}

/**
 * Display prediction results
 */
function displayResults(result) {
    // Update stats
    currentStatusDiv.textContent = result.is_fall ? '🚨 FALL' : '✅ Safe';
    currentStatusDiv.style.color = result.is_fall ? '#ff4757' : '#2ed573';
    
    fallProbabilityDiv.textContent = `${(result.fall_probability * 100).toFixed(1)}%`;
    confidenceDiv.textContent = `${(result.confidence * 100).toFixed(1)}%`;
    peakAccelerationDiv.textContent = `${result.max_acceleration.toFixed(2)} m/s²`;
    
    // Build results HTML
    const html = `
        <div class="${result.alert ? 'result-alert' : ''}">
            <div class="result-item">
                <span class="result-key">Fall Detected:</span>
                <span class="result-value">${result.is_fall ? '🚨 YES' : '✅ NO'}</span>
            </div>
            <div class="result-item">
                <span class="result-key">Fall Probability:</span>
                <span class="result-value">${(result.fall_probability * 100).toFixed(2)}%</span>
            </div>
            <div class="result-item">
                <span class="result-key">No-Fall Probability:</span>
                <span class="result-value">${(result.no_fall_probability * 100).toFixed(2)}%</span>
            </div>
            <div class="result-item">
                <span class="result-key">Confidence:</span>
                <span class="result-value">${(result.confidence * 100).toFixed(2)}%</span>
            </div>
            <div class="result-item">
                <span class="result-key">Max Acceleration:</span>
                <span class="result-value">${result.max_acceleration.toFixed(2)} m/s²</span>
            </div>
            <div class="result-item">
                <span class="result-key">Mean Acceleration:</span>
                <span class="result-value">${result.mean_acceleration.toFixed(2)} m/s²</span>
            </div>
            <div class="result-item">
                <span class="result-key">Peak Jerk:</span>
                <span class="result-value">${result.peak_jerk.toFixed(2)} m/s³</span>
            </div>
            <div class="result-item">
                <span class="result-key">Timestamp:</span>
                <span class="result-value">${new Date(result.timestamp).toLocaleString()}</span>
            </div>
            ${result.alert ? '<div class="alert-text">⚠️ ALERT: Fall detected with high confidence!</div>' : ''}
        </div>
    `;
    
    resultsContainer.innerHTML = html;
}

/**
 * Display acceleration chart
 */
function displayChart(accelerometerData) {
    // Calculate magnitude
    const magnitude = accelerometerData.map(([x, y, z]) => 
        Math.sqrt(x*x + y*y + z*z)
    );
    
    const labels = magnitude.map((_, i) => i);
    
    // Destroy existing chart if any
    if (chart) {
        chart.destroy();
    }
    
    // Create new chart
    chart = new Chart(accelerationCanvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Acceleration Magnitude',
                data: magnitude,
                borderColor: '#ff4757',
                backgroundColor: 'rgba(255, 71, 87, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 0,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        font: {
                            size: 12
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Acceleration (m/s²)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time Sample'
                    }
                }
            }
        }
    });
}

/**
 * Handle load demo data button
 */
function handleLoadDemo() {
    generateDemoData();
    addLog('Demo data loaded', 'success');
}

/**
 * Generate demo accelerometer data
 */
function generateDemoData() {
    // Generate 60 samples of accelerometer data
    const data = [];
    
    // Mix of normal and fall-like data
    for (let i = 0; i < 60; i++) {
        const t = i / 20; // Time in seconds
        
        // Fall simulation: sharp acceleration spike in middle
        let x, y, z;
        if (i > 25 && i < 40) {
            // Simulated fall: high acceleration
            x = (Math.random() - 0.5) * 5;
            y = (Math.random() - 0.5) * 5;
            z = -8 + Math.random() * 2; // Gravity + impact
        } else {
            // Normal movement: low acceleration
            x = (Math.random() - 0.5) * 1;
            y = (Math.random() - 0.5) * 1;
            z = -9.8 + (Math.random() - 0.5) * 0.5; // Gravity
        }
        
        data.push([x, y, z]);
    }
    
    dataInput.value = JSON.stringify(data, null, 2);
}

/**
 * Handle clear button
 */
function handleClear() {
    dataInput.value = '';
    resultsContainer.innerHTML = '<p class="placeholder">Results will appear here...</p>';
    currentStatusDiv.textContent = 'Ready';
    currentStatusDiv.style.color = '';
    fallProbabilityDiv.textContent = '0%';
    confidenceDiv.textContent = '0%';
    peakAccelerationDiv.textContent = '0 m/s²';
    
    if (chart) {
        chart.destroy();
        chart = null;
    }
    
    addLog('Data cleared', 'info');
}

/**
 * Handle paste event
 */
function handlePaste(e) {
    // Let paste happen naturally
    setTimeout(() => {
        try {
            const data = JSON.parse(dataInput.value);
            addLog(`Pasted ${data.length} samples`, 'success');
        } catch (e) {
            // Invalid JSON, will be caught during prediction
        }
    }, 10);
}

/**
 * Add log entry
 */
function addLog(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('p');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${timestamp}] ${message}`;
    
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
    
    // Keep only last 50 entries
    while (logContainer.children.length > 50) {
        logContainer.removeChild(logContainer.firstChild);
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', init);