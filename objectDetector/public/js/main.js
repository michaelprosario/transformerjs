class ObjectDetectionApp {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
        this.checkServerStatus();
    }

    initializeElements() {
        this.form = document.getElementById('uploadForm');
        this.imageInput = document.getElementById('imageInput');
        this.fileLabel = document.querySelector('.file-input-label');
        this.detectBtn = document.querySelector('.detect-btn');
        this.previewSection = document.getElementById('previewSection');
        this.imagePreview = document.getElementById('imagePreview');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');
        this.detectionCount = document.getElementById('detectionCount');
        this.resultImage = document.getElementById('resultImage');
        this.detectionCanvas = document.getElementById('detectionCanvas');
        this.detectionsList = document.getElementById('detectionsList');
        this.errorText = document.getElementById('errorText');
        this.statusText = document.getElementById('statusText');
        this.statusDot = document.getElementById('statusDot');
    }

    setupEventListeners() {
        this.imageInput.addEventListener('change', (e) => this.handleFileSelect(e));
        this.form.addEventListener('submit', (e) => this.handleFormSubmit(e));
    }

    async checkServerStatus() {
        try {
            this.updateStatus('Checking server status...', 'loading');
            const response = await fetch('/health');
            const data = await response.json();
            
            if (data.status === 'ok') {
                if (data.modelLoaded) {
                    this.updateStatus('Server online - Model ready', 'online');
                } else {
                    this.updateStatus('Server online - Loading model...', 'loading');
                    // Check again in 3 seconds if model is still loading
                    setTimeout(() => this.checkServerStatus(), 3000);
                }
            } else {
                this.updateStatus('Server error', 'offline');
            }
        } catch (error) {
            this.updateStatus('Server offline', 'offline');
            console.error('Status check failed:', error);
        }
    }

    updateStatus(text, status) {
        this.statusText.textContent = text;
        this.statusDot.className = `status-dot ${status}`;
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        
        if (file) {
            // Update UI to show file is selected
            this.fileLabel.classList.add('file-selected');
            this.fileLabel.querySelector('.upload-text').textContent = file.name;
            this.detectBtn.disabled = false;

            // Show image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                this.imagePreview.src = e.target.result;
                this.previewSection.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Hide previous results
            this.hideAllSections();
        }
    }

    async handleFormSubmit(event) {
        event.preventDefault();
        
        const file = this.imageInput.files[0];
        if (!file) return;

        this.showLoading();
        
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.showResults(data);
            } else {
                this.showError(data.error || 'Detection failed');
            }
        } catch (error) {
            console.error('Detection error:', error);
            this.showError('Network error. Please check your connection and try again.');
        }
    }

    showLoading() {
        this.hideAllSections();
        this.loadingSection.style.display = 'block';
        this.detectBtn.classList.add('loading');
        this.detectBtn.disabled = true;
    }

    showResults(data) {
        this.hideAllSections();
        this.resultsSection.style.display = 'block';
        
        // Update detection count
        this.detectionCount.textContent = data.totalDetections;
        
        // Show result image
        this.resultImage.src = data.image;
        this.resultImage.onload = () => {
            this.drawDetections(data.detections);
        };
        
        // Show detections list
        this.populateDetectionsList(data.detections);
        
        // Reset button
        this.detectBtn.classList.remove('loading');
        this.detectBtn.disabled = false;
    }

    drawDetections(detections) {
        const canvas = this.detectionCanvas;
        const ctx = canvas.getContext('2d');
        const img = this.resultImage;
        
        // Set canvas size to match image
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        canvas.style.width = img.offsetWidth + 'px';
        canvas.style.height = img.offsetHeight + 'px';
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw bounding boxes
        detections.forEach((detection, index) => {
            const { box } = detection;
            const color = this.getColorForIndex(index);
            
            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.strokeRect(box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin);
            
            // Draw label background
            const labelText = `${detection.label} (${Math.round(detection.score * 100)}%)`;
            ctx.font = '16px Arial';
            const textMetrics = ctx.measureText(labelText);
            const textWidth = textMetrics.width;
            const textHeight = 20;
            
            ctx.fillStyle = color;
            ctx.fillRect(box.xmin, box.ymin - textHeight - 5, textWidth + 10, textHeight + 5);
            
            // Draw label text
            ctx.fillStyle = 'white';
            ctx.fillText(labelText, box.xmin + 5, box.ymin - 8);
        });
    }

    getColorForIndex(index) {
        const colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57',
            '#ff9ff3f4', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9ff3',
            '#ffb142', '#ff6348', '#1dd1a1', '#feca57', '#48dbfb'
        ];
        return colors[index % colors.length];
    }

    populateDetectionsList(detections) {
        this.detectionsList.innerHTML = '';
        
        if (detections.length === 0) {
            const li = document.createElement('li');
            li.innerHTML = '<span class="detection-label">No objects detected</span>';
            this.detectionsList.appendChild(li);
            return;
        }
        
        detections.forEach((detection) => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="detection-label">${detection.label}</span>
                <span class="detection-confidence">${Math.round(detection.score * 100)}%</span>
            `;
            this.detectionsList.appendChild(li);
        });
    }

    showError(message) {
        this.hideAllSections();
        this.errorSection.style.display = 'block';
        this.errorText.textContent = message;
        
        // Reset button
        this.detectBtn.classList.remove('loading');
        this.detectBtn.disabled = false;
    }

    hideAllSections() {
        this.loadingSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'none';
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ObjectDetectionApp();
});
