// EmergencyAI - Backend API Integration

// Use our proxy to avoid CORS issues
const API_URL = '/api/direct';

// Function to generate audio using MiniMax API
async function generateEmergencyAudio(emergencyType, customText = null) {
    const emergencyData = {
        medical: {
            title: "Medical Emergency Response",
            script: "This is Emergency AI. I'll guide you through this medical emergency. First, check if the person is responsive. If they're unresponsive, call emergency services immediately. Check for breathing by looking for chest movement and listening for breath sounds. If not breathing normally, begin CPR if trained. Push hard and fast in the center of the chest. If an AED is available, use it following the instructions. Continue CPR until help arrives or the person shows signs of life. Stay calm and focused."
        },
        fire: {
            title: "Fire Emergency Response",
            script: "This is Emergency AI. I'll help you through this fire emergency. First, activate the nearest fire alarm and call emergency services. If the fire is small and contained, use a fire extinguisher if you're trained to do so. Evacuate the building immediately and do not use elevators. If there's smoke, stay low to the ground where air is clearer. Feel doors before opening them - if a door is hot, find another exit route. Once outside, move to your designated assembly point and wait for emergency services. Your safety is the priority."
        },
        security: {
            title: "Security Threat Response",
            script: "This is Emergency AI. I'll guide you through this security threat. If possible, evacuate the area immediately. If evacuation is not possible, find a place to hide where you won't be trapped. Lock and barricade doors, close blinds, and turn off lights. Silence your cell phone and remain quiet. Call emergency services when it's safe to do so. When law enforcement arrives, keep your hands visible and follow all instructions. Stay calm and focused on your safety."
        },
        natural: {
            title: "Natural Disaster Response",
            script: "This is Emergency AI. I'll help you respond to this natural disaster. First, identify what type of disaster is occurring - earthquake, flood, tornado, or other. Move to the safest location based on the disaster type. For earthquakes: Drop, Cover, and Hold On under sturdy furniture. For floods: Move to higher ground immediately. For tornadoes: Go to a basement or interior room on the lowest floor. Stay informed through emergency broadcasts and follow official instructions. Your preparedness can save lives."
        }
    };

    const text = customText || emergencyData[emergencyType].script;
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                tool: 'text_to_audio',
                parameters: {
                    text: text,
                    voice_id: 'Grinch', // Using a calm, authoritative voice
                    model: 'speech-02-hd',
                    speed: 0.9 // Slightly slower for clarity
                }
            })
        });
        
        const data = await response.json();
        console.log('Audio generation response:', data);
        
        if (data.success && data.result && data.result.audio_path) {
            console.log('Raw audio path from API:', data.result.audio_path);
            
            // Convert the audio path to a URL that our server can serve
            // The MiniMax API returns paths like './output/file.mp3' or '../output/file.mp3'
            // We need to convert this to '/output/file.mp3' for our server to serve it
            let audioPath = data.result.audio_path;
            
            // Extract just the filename from the path
            const filename = audioPath.split('/').pop();
            
            // Ensure the filename is valid
            if (!filename || filename === '') {
                console.error('Invalid filename extracted from path:', audioPath);
                throw new Error('Invalid audio filename');
            }
            
            // Create the server path
            const serverPath = `/output/${filename}`;
            
            console.log('Converted audio path for server:', serverPath);
            
            // Verify the file exists by making a HEAD request
            try {
                const checkResponse = await fetch(serverPath, { method: 'HEAD' });
                if (!checkResponse.ok) {
                    console.warn(`Audio file not found at ${serverPath}, status: ${checkResponse.status}`);
                }
            } catch (error) {
                console.warn('Error checking audio file:', error);
            }
            
            return {
                success: true,
                audioPath: serverPath,
                originalPath: data.result.audio_path,
                metadata: data.metadata
            };
        } else {
            console.error('Error generating audio:', data.error || 'Unknown error');
            return {
                success: false,
                error: data.error || 'Failed to generate audio guidance'
            };
        }
    } catch (error) {
        console.error('API request failed:', error);
        return {
            success: false,
            error: 'Network error or API unavailable'
        };
    }
}

// Function to generate supportive images using MiniMax API
async function generateSupportImage(emergencyType) {
    const prompts = {
        medical: "Clear medical emergency first aid instructions diagram, CPR technique, recovery position, professional medical illustration",
        fire: "Fire evacuation procedure diagram, fire safety illustration, emergency exit plan, professional safety diagram",
        security: "Security lockdown procedure illustration, shelter in place diagram, professional safety instructions",
        natural: "Natural disaster safety procedures, earthquake safety position, flood evacuation route, professional emergency diagram"
    };

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                tool: 'text_to_image',
                parameters: {
                    prompt: prompts[emergencyType],
                    negative_prompt: "blurry, low quality, distorted, confusing, scary, graphic",
                    width: 512,
                    height: 512
                }
            })
        });
        
        const data = await response.json();
        console.log('Image generation response:', data);
        
        if (data.success && data.result) {
            return {
                success: true,
                imagePath: data.result.image_path || data.result.image_url
            };
        } else {
            console.error('Error generating image:', data.error || 'Unknown error');
            return {
                success: false,
                error: data.error || 'Failed to generate supportive image'
            };
        }
    } catch (error) {
        console.error('API request failed:', error);
        return {
            success: false,
            error: 'Network error or API unavailable'
        };
    }
}

// Function to check API health
async function checkApiHealth() {
    try {
        // First check our own server's health
        const localResponse = await fetch('/health');
        const localData = await localResponse.json();
        console.log('Local server health status:', localData);
        
        // Then check the MiniMax API health through our proxy
        const apiResponse = await fetch('/api/health');
        const apiData = await apiResponse.json();
        console.log('MiniMax API health status:', apiData);
        
        return apiData.status === 'healthy';
    } catch (error) {
        console.error('Health check failed:', error);
        return false;
    }
}

// Export functions for use in the main app
window.EmergencyAI = {
    generateAudio: generateEmergencyAudio,
    generateImage: generateSupportImage,
    checkApiHealth: checkApiHealth
};
