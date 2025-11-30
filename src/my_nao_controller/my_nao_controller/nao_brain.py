"""
nao_brain.py - AI-Powered Conversation Controller for NAO Robot

This module implements the main AI pipeline for the persona-aware NAO robot.
It handles the complete conversation cycle:

1. Voice Activity Detection (VAD) - Silero VAD for speech detection
2. Speech-to-Text (STT) - Nvidia Riva for transcription  
3. LLM Processing - Nvidia NIM (Llama) for persona-aware responses
4. Action Selection - Semantic search to map intent to robot gestures
5. Text-to-Speech (TTS) - Nvidia Riva for speech synthesis

The robot maintains conversation memory and supports multiple personas
(e.g., polite teacher, angry cab driver) that affect both speech style
and gesture selection.

Author: Vashu Chauhan
"""

import os
import sys
import time
import subprocess
import torch
import numpy as np
import pickle
import threading
import sounddevice as sd
from scipy.io.wavfile import write, read
from openai import OpenAI
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Semantic Search Imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import configuration from parent/config folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'config'))
import config

# Import Personas
from my_nao_controller.personas import PERSONAS

# Validate configuration on startup
try:
    config.validate_config()
    print(config.get_config_summary())
except ValueError as e:
    print(f"[CRITICAL ERROR] Configuration validation failed:")
    print(e)
    sys.exit(1)


class ActionSelector:
    """
    Semantic search engine for mapping LLM intents to robot actions.
    
    Uses sentence embeddings (all-MiniLM-L6-v2) to find the closest
    matching action based on cosine similarity between the LLM's
    stated intent and pre-computed action description embeddings.
    
    Attributes:
        model: SentenceTransformer model for encoding text
        keys (list): Action names from the vocabulary
        db_embeddings (ndarray): Pre-computed embeddings for action descriptions
    
    Example:
        selector = ActionSelector()
        action = selector.search("greeting someone warmly")
        # Returns: "wave_right_hand" (if best match)
    """
    
    def __init__(self):
        """
        Initialize the action selector with embedding model and action database.
        
        Loads the pre-computed action embeddings from the pickle file.
        If the file doesn't exist, initializes with empty lists (will fail on search).
        """
        print(f"   [Init] Loading Embedding Model ({config.EMBEDDING_MODEL})...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        
        if not os.path.exists(config.EMBEDDING_FILE):
            print(f"CRITICAL ERROR: {config.EMBEDDING_FILE} not found.")
            self.keys = []
            self.db_embeddings = []
        else:
            with open(config.EMBEDDING_FILE, "rb") as f:
                data = pickle.load(f)
                self.keys = data["keys"]
                self.db_embeddings = data["embeddings"]
            print(f"   [Init] Loaded {len(self.keys)} actions.")

    def search(self, query_text):
        """
        Find the best matching action for a given intent query.
        
        Encodes the query text and computes cosine similarity against
        all action embeddings. Returns the action with highest similarity,
        or the default action if confidence is below threshold.
        
        Args:
            query_text (str): The intent text from LLM (e.g., "greeting", "explaining")
            
        Returns:
            str: The name of the best matching action (e.g., "wave_right_hand")
        """
        t0 = time.time()
        # Encode the query intent into an embedding vector
        query_vec = self.model.encode([query_text])
        # Compute similarity scores against all action embeddings
        scores = cosine_similarity(query_vec, self.db_embeddings)[0]
        best_idx = np.argmax(scores)
        best_action = self.keys[best_idx]
        confidence = scores[best_idx]
        
        if config.VERBOSE_LOGGING:
            print(f"\n   [Action Mapping]")
            print(f"   Query Intent:   '{query_text}'")
            print(f"   Mapped To:      '{best_action}'")
            print(f"   Confidence:     {confidence:.4f}")
        
        if confidence < config.ACTION_CONFIDENCE_THRESHOLD:
            if config.VERBOSE_LOGGING:
                print(f"   (! Weak Match ! Defaulting to '{config.DEFAULT_ACTION}')")
            return config.DEFAULT_ACTION
        return best_action


class NaoBrain(Node):
    """
    Main ROS 2 node implementing the AI conversation pipeline.
    
    Orchestrates the full interaction cycle: listening for speech,
    transcribing, generating persona-aware responses, selecting
    appropriate gestures, and speaking the response.
    
    Attributes:
        publisher_: ROS 2 publisher for /perform_action topic
        conversation_history (list): Memory of past exchanges for context
        persona (dict): Current personality profile with speaking style
        selector (ActionSelector): Semantic search for action mapping
        model: Silero VAD model for voice activity detection
        VADIterator: Iterator class for streaming VAD
    
    Supported Personas:
        - polite_teacher: Patient, educational responses
        - polite_receptionist: Professional, courteous help
        - angry_cab_driver: Gruff, impatient exchanges
    """
    
    def __init__(self):
        """
        Initialize the NaoBrain node with all AI components.
        
        Sets up:
        - ROS 2 publisher for action commands
        - Conversation memory system
        - Selected persona configuration  
        - Action selector (semantic search)
        - Voice Activity Detection model (Silero)
        
        Raises:
            ValueError: If configured persona ID is not found
        """
        super().__init__(config.ROS_NODE_NAME)
        # Create publisher for sending action commands to nao_driver
        self.publisher_ = self.create_publisher(
            String, 
            config.ROS_ACTION_TOPIC, 
            config.ROS_QUEUE_SIZE
        )
        
        # Initialize conversation memory
        self.conversation_history = [] if config.ENABLE_CONVERSATION_MEMORY else None
        
        # Load Persona
        if config.CURRENT_PERSONA_ID not in PERSONAS:
            self.get_logger().error(f"Invalid persona ID: {config.CURRENT_PERSONA_ID}")
            self.get_logger().info(f"Available personas: {list(PERSONAS.keys())}")
            raise ValueError(f"Persona '{config.CURRENT_PERSONA_ID}' not found")
            
        self.persona = PERSONAS[config.CURRENT_PERSONA_ID]
        self.get_logger().info(f"Loaded Persona: {self.persona['name']} ({self.persona['role']})")
        
        self.selector = ActionSelector()
        print("   [Init] Loading VAD Model...")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad', 
            force_reload=config.FORCE_MODEL_RELOAD, 
            verbose=config.VERBOSE_LOGGING
        )
        (self.get_speech_timestamps, _, _, self.VADIterator, _) = utils
        
    def construct_system_prompt(self):
        """Build a persona-specific system prompt"""
        p = self.persona
        
        # Build example phrases section
        examples = "\n".join([f"- {phrase}" for phrase in p['example_phrases']])
        
        memory_instruction = ""
        if config.ENABLE_CONVERSATION_MEMORY:
            memory_instruction = "\n7. Remember the conversation context and refer back to previous exchanges when relevant"
        
        prompt = f"""You are {p['name']}, a Nao Robot with the role of {p['role']}.

PERSONALITY: You are {p['personality']}.

SPEAKING STYLE: {p['speaking_style']}

EXAMPLE PHRASES YOU MIGHT USE:
{examples}

CRITICAL RESPONSE FORMAT:
1. First, determine the user's intent in 2-4 words (like: greeting, asking question, requesting action, thanking, etc)
2. Then respond with speech that matches your persona
3. Format: intent | speech
4. IMPORTANT: Use NO PUNCTUATION in your speech (no periods, commas, exclamation marks, question marks)
5. Keep responses under 30 words
6. Stay in character at all times{memory_instruction}

Example responses:
- "greeting | Hey there what do you want"
- "asking directions | Look I dont know just use your phone pal"
- "thanking | Yeah yeah sure whatever"
"""
        return prompt

    def add_to_conversation(self, role, content):
        """Add a message to conversation history"""
        if not config.ENABLE_CONVERSATION_MEMORY:
            return
            
        self.conversation_history.append({"role": role, "content": content})
        
        # Trim history if it exceeds max length
        max_messages = (config.MAX_CONVERSATION_HISTORY * 2)
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]
        
        if config.VERBOSE_LOGGING:
            print(f"    [Memory] Conversation history: {len(self.conversation_history)} messages")

    def get_conversation_summary(self):
        """Get a brief summary of conversation for logging"""
        if not config.ENABLE_CONVERSATION_MEMORY:
            return "Memory disabled"
        if len(self.conversation_history) == 0:
            return "No history"
        return f"{len(self.conversation_history)} messages (last {min(config.MAX_CONVERSATION_HISTORY, len(self.conversation_history)//2)} exchanges)"

    def run_cycle(self):
        """
        Execute one complete conversation cycle.
        
        Pipeline steps:
        1. Record user audio with VAD
        2. Transcribe audio to text (STT)
        3. Generate persona response (LLM) -> intent + speech
        4. Map intent to action (semantic search)
        5. Synthesize speech (TTS)
        6. Execute action + play audio (parallel or sequential)
        7. Return to neutral pose
        
        Supports parallel execution of TTS and action search for
        lower latency when config.ENABLE_PARALLEL_EXECUTION is True.
        """
        print("\n" + config.LOG_SEPARATOR)
        print(f"   PERSONA: {self.persona['name']} ({self.persona['role']})")
        print(f"   MEMORY: {self.get_conversation_summary()}")
        print(config.LOG_SEPARATOR)
        
        if self.record_user_audio():
            user_text = self.transcribe_audio()
            
            # Add user message to conversation history
            self.add_to_conversation("user", user_text)
            
            # Generate response with persona-specific prompt and conversation history
            llm_intent, speech = self.generate_response()
            
            # Add assistant response to conversation history
            self.add_to_conversation("assistant", f"{llm_intent} | {speech}")
            
            if config.ENABLE_PARALLEL_EXECUTION:
                print(f"\n   [Parallel Execution Started]")
                results = {}
                
                def do_tts():
                    results['tts_success'] = self.generate_audio_file(speech)
                
                def do_search():
                    results['action'] = self.selector.search(llm_intent)
                    
                t1 = threading.Thread(target=do_tts)
                t2 = threading.Thread(target=do_search)
                
                t1.start()
                t2.start()
                
                # Wait for action search first
                t2.join()
                print(f"   --> Publishing Action: {results['action']}")
                self.publish_action(results['action'])
                
                # Wait for TTS
                t1.join()
                if results.get('tts_success', False):
                    self.play_audio_file()
            else:
                # Sequential execution
                action = self.selector.search(llm_intent)
                print(f"   --> Publishing Action: {action}")
                self.publish_action(action)
                
                if self.generate_audio_file(speech):
                    self.play_audio_file()

            # Reset to neutral
            self.publish_action(config.DEFAULT_ACTION)
        

    def publish_action(self, action_name):
        """
        Publish an action command to the robot driver.
        
        Args:
            action_name (str): Name of the action to perform
                              (e.g., "wave_right_hand", "stand_neutral")
        """
        msg = String()
        msg.data = action_name
        self.publisher_.publish(msg)

    # --- VAD (Voice Activity Detection) ---
    def record_user_audio(self):
        """
        Record audio from microphone until silence is detected.
        
        Uses Silero VAD to detect speech onset and offset.
        Recording stops after SILENCE_THRESHOLD seconds of silence
        following detected speech.
        
        Returns:
            bool: True if audio was recorded successfully
        
        Side Effects:
            Writes recorded audio to config.AUDIO_INPUT file
        """
        print(f"\n[1] Listening...")
        # Create fresh VAD iterator for this recording session
        vad_iterator = self.VADIterator(self.model)
        buffer = []  # Accumulate audio chunks
        is_speaking = False  # Track if user has started speaking
        silence_start = None  # Timestamp when silence began
        
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE, 
            channels=config.CHANNELS, 
            dtype='float32', 
            blocksize=config.BLOCK_SIZE
        ) as stream:
            while True:
                chunk, _ = stream.read(config.BLOCK_SIZE)
                audio_tensor = torch.from_numpy(chunk.flatten())
                speech_dict = vad_iterator(audio_tensor, return_seconds=True)
                buffer.append(chunk)
                
                if speech_dict:
                    if "start" in speech_dict and not is_speaking:
                        print("    --> Speech detected...")
                        is_speaking = True
                    silence_start = None
                elif is_speaking:
                    if silence_start is None: 
                        silence_start = time.time()
                    if (time.time() - silence_start) > config.SILENCE_THRESHOLD:
                        print("    --> Silence detected. Processing...")
                        break
        
        full_audio = np.concatenate(buffer)
        
        # Normalize audio if enabled
        if config.NORMALIZE_AUDIO:
            max_val = np.max(np.abs(full_audio))
            if max_val > 0: 
                full_audio = full_audio / max_val
        
        full_audio_int16 = (full_audio * 32767).astype(np.int16)
        write(config.AUDIO_INPUT, config.SAMPLE_RATE, full_audio_int16)
        return True

    # --- STT (Speech-to-Text) ---
    def transcribe_audio(self):
        """
        Transcribe recorded audio to text using Nvidia Riva STT.
        
        Calls the Riva transcription script as a subprocess and
        parses the transcript from the output.
        
        Returns:
            str: Transcribed text, or FALLBACK_TRANSCRIPT on error
        """
        print(f"\n[2] STT ({config.STT_MODEL})")
        t0 = time.time()
        
        cmd = [
            "python3", config.STT_SCRIPT, 
            "--server", config.RIVA_SERVER,
            "--use-ssl" if config.USE_SSL else "--no-ssl",
            "--metadata", "function-id", config.STT_FUNCTION_ID, 
            "--metadata", "authorization", f"Bearer {config.NVIDIA_API_KEY}", 
            "--language-code", config.STT_LANGUAGE_CODE, 
            "--input-file", config.AUDIO_INPUT
        ]
        
        transcript = None
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.STT_TIMEOUT)
            
            # Parse transcript from output
            for line in result.stdout.split('\n'):
                line = line.strip()
                if "Final transcript:" in line:
                    transcript = line.split("Final transcript:", 1)[1].strip()
                    break
                if line.startswith("Transcript:"):
                    transcript = line.split("Transcript:", 1)[1].strip()
                    break
            
            if not transcript and '"transcript":' in result.stdout:
                 for line in result.stdout.split('\n'):
                     if '"transcript":' in line:
                         parts = line.split('"transcript":')
                         if len(parts) > 1:
                             transcript = parts[1].strip().strip('",')
                             break
                             
            if not transcript or transcript == "":
                if config.VERBOSE_LOGGING:
                    print(f"    [WARNING] STT Failed to parse. Raw output:\n{result.stdout}")
                transcript = config.FALLBACK_TRANSCRIPT
                
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] STT timeout after {config.STT_TIMEOUT}s")
            transcript = config.FALLBACK_TRANSCRIPT
        except Exception as e:
            print(f"    [ERROR] STT subprocess failed: {e}")
            transcript = config.FALLBACK_TRANSCRIPT

        print(f"    Transcript: '{transcript}'")
        if config.LOG_LATENCY:
            print(f"    Latency:    {time.time() - t0:.4f}s")
        return transcript

    # --- LLM (Large Language Model) ---
    def generate_response(self):
        """
        Generate a persona-aware response using the LLM.
        
        Constructs a system prompt with persona details, includes
        conversation history for context, and requests a response
        in the format: "intent | speech"
        
        Returns:
            tuple: (intent, speech) where:
                - intent (str): Short action descriptor (e.g., "greeting")
                - speech (str): Text for TTS to speak
        
        Note:
            Uses streaming for real-time output if config.LLM_STREAMING is True
        """
        print(f"\n[3] LLM ({config.LLM_MODEL.split('/')[-1]}) - Persona: {self.persona['name']}")
        if config.ENABLE_CONVERSATION_MEMORY:
            print(f"    Context: {len(self.conversation_history)} messages in history")
        
        t0 = time.time()
        client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.NVIDIA_API_KEY)
        
        intent = config.FALLBACK_INTENT
        speech = config.FALLBACK_SPEECH
        
        try:
            # Build messages with system prompt + conversation history
            messages = [
                {"role": "system", "content": self.construct_system_prompt()}
            ]
            
            if config.ENABLE_CONVERSATION_MEMORY:
                messages.extend(self.conversation_history)
            else:
                # If memory disabled, only use the last user message
                if self.conversation_history:
                    messages.append(self.conversation_history[-1])
            
            completion = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                stream=config.LLM_STREAMING
            )
            
            if config.LLM_STREAMING:
                print("    Stream: ", end="", flush=True)
                full_resp = ""
                first_token = False
                for chunk in completion:
                    c = chunk.choices[0].delta.content
                    if c:
                        if not first_token and config.LOG_LATENCY:
                            print(f"(TTFT: {time.time()-t0:.2f}s) ", end="")
                            first_token = True
                        print(c, end="", flush=True)
                        full_resp += c
                print("")
            else:
                full_resp = completion.choices[0].message.content
            
            if config.LOG_LATENCY:
                print(f"    Total Latency: {time.time()-t0:.4f}s")
            
            # Parse response
            if "|" in full_resp:
                parts = full_resp.split("|", 1)
                intent = parts[0].strip()
                speech = parts[1].strip()
            else:
                speech = full_resp.strip()
                intent = config.FALLBACK_INTENT
                
            if config.VERBOSE_LOGGING:
                print(f"    Parsed Intent: '{intent}'")
                print(f"    Parsed Speech: '{speech}'")
            
        except Exception as e:
            print(f"    [ERROR] LLM failed: {e}")

        return intent, speech

    # --- TTS (Text-to-Speech) ---
    def generate_audio_file(self, speech_text):
        """
        Generate speech audio from text using Nvidia Riva TTS.
        
        Uses the persona's TTS settings (voice, language) to
        synthesize natural-sounding speech.
        
        Args:
            speech_text (str): The text to convert to speech
            
        Returns:
            bool: True if audio file was generated successfully
        
        Side Effects:
            Writes audio to config.AUDIO_OUTPUT file
        """
        settings = self.persona["tts_settings"]
        voice_name = f"Magpie-Multilingual.{settings['language']}.{settings['voice']}"
        
        print(f"\n[4] TTS ({config.TTS_MODEL}) - Voice: {voice_name}")
        if config.VERBOSE_LOGGING:
            print(f"    Speech Text: '{speech_text}'")
        
        t0 = time.time()
        
        cmd = [
            "python3", config.TTS_SCRIPT, 
            "--server", config.RIVA_SERVER, 
            "--use-ssl" if config.USE_SSL else "--no-ssl",
            "--metadata", "function-id", config.TTS_FUNCTION_ID, 
            "--metadata", "authorization", f"Bearer {config.NVIDIA_API_KEY}", 
            "--language-code", settings['language'], 
            "--text", speech_text, 
            "--voice", config.TTS_VOICE,
            "--output", config.AUDIO_OUTPUT
        ]
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True, 
                timeout=config.TTS_TIMEOUT
            )
            if config.LOG_LATENCY:
                print(f"    Latency: {time.time()-t0:.4f}s")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] TTS timeout after {config.TTS_TIMEOUT}s")
            return False
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] TTS failed with code {e.returncode}")
            if config.VERBOSE_LOGGING:
                print(f"    STDOUT: {e.stdout}")
                print(f"    STDERR: {e.stderr}")
            return False
        except Exception as e:
            print(f"    [ERROR] TTS failed: {e}")
            return False

    def play_audio_file(self):
        """
        Play the generated speech audio file through speakers.
        
        Reads the audio file from config.AUDIO_OUTPUT and plays
        it synchronously (blocks until playback completes).
        """
        try:
            fs, data = read(config.AUDIO_OUTPUT)
            duration = len(data) / fs
            print(f"    --> Playing Audio ({duration:.2f}s)...")
            sd.play(data, fs)
            sd.wait()  # Block until playback finishes
        except Exception as e: 
            self.get_logger().error(f"Audio playback failed: {e}")


def main(args=None):
    """
    Entry point for the NAO Brain node.
    
    Initializes ROS 2, creates the NaoBrain node, and runs the
    conversation loop indefinitely until interrupted.
    
    Args:
        args: Command-line arguments (passed to rclpy.init)
    
    Usage:
        ros2 run my_nao_controller nao_brain
    """
    rclpy.init(args=args)
    
    try:
        node = NaoBrain()
        if config.ENABLE_CONVERSATION_MEMORY:
            print(f"\n[Memory System] Initialized - conversation will be remembered across cycles")
            print(f"[Memory System] Max history: {config.MAX_CONVERSATION_HISTORY} exchanges")
        else:
            print(f"\n[Memory System] Disabled - each cycle is independent")
        
        while rclpy.ok(): 
            node.run_cycle()
    except KeyboardInterrupt: 
        print("\n[Shutdown] Keyboard interrupt received")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()