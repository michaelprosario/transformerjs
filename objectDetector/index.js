const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { pipeline } = require('@xenova/transformers');
const sharp = require('sharp');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, uploadsDir);
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: (req, file, cb) => {
        const allowedTypes = /jpeg|jpg|png|gif|webp/;
        const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
        const mimetype = allowedTypes.test(file.mimetype);
        
        if (mimetype && extname) {
            return cb(null, true);
        } else {
            cb(new Error('Only image files are allowed!'));
        }
    },
    limits: {
        fileSize: 5 * 1024 * 1024 // 5MB limit
    }
});

// Initialize the object detection pipeline
let detector = null;

async function initializeDetector() {
    try {
        console.log('Loading object detection model...');
        detector = await pipeline('object-detection', 'Xenova/detr-resnet-50', {
            quantized: false,
        });
        console.log('Object detection model loaded successfully!');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

// Initialize the model when the server starts
initializeDetector();

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Object detection endpoint
app.post('/detect', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No image file provided' });
        }

        if (!detector) {
            return res.status(503).json({ error: 'Model is still loading. Please try again in a moment.' });
        }

        const imagePath = req.file.path;
        console.log('Processing image:', imagePath);

        // Run object detection directly on the image file
        const results = await detector(imagePath);
        
        console.log('Detection results:', results);

        // Process the image with Sharp for frontend display
        const imageBuffer = await sharp(imagePath)
            .resize(800, 600, { fit: 'inside', withoutEnlargement: true })
            .jpeg()
            .toBuffer();

        // Convert image to base64 for frontend display
        const base64Image = imageBuffer.toString('base64');

        // Filter results by confidence threshold
        const filteredResults = results.filter(result => result.score > 0.3);

        // Clean up uploaded file
        fs.unlinkSync(imagePath);

        res.json({
            success: true,
            detections: filteredResults,
            image: `data:image/jpeg;base64,${base64Image}`,
            totalDetections: filteredResults.length
        });

    } catch (error) {
        console.error('Error during detection:', error);
        
        // Clean up uploaded file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        
        res.status(500).json({ 
            error: 'Error processing image', 
            details: error.message 
        });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        modelLoaded: detector !== null,
        timestamp: new Date().toISOString()
    });
});

// Error handling middleware
app.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({ error: 'File size too large. Maximum size is 5MB.' });
        }
    }
    
    console.error('Unhandled error:', error);
    res.status(500).json({ error: 'Internal server error' });
});

app.listen(PORT, () => {
    console.log(`Object Detection Server running on http://localhost:${PORT}`);
    console.log('Waiting for model to load...');
});