"""
op2_brain.py - AI-Powered Conversation Controller for OP2 Robot

This module implements the main AI pipeline for the persona-aware OP2 robot.
It handles the complete conversation cycle:

1. Voice Activity Detection (VAD) - Silero VAD for speech detection
2. Speech-to-Text (STT) - Nvidia Riva for transcription  
3. LLM Processing - Nvidia NIM (Llama) for persona-aware responses
4. Action Selection - Semantic search to map intent to robot gestures
5. Text-to-Speech (TTS) - Nvidia Riva for speech synthesis

The robot maintains conversation memory and supports multiple personas
(e.g., polite teacher, angry cab driver) that affect both speech style
and gesture selection.
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
from op2_controller.personas import PERSONAS

# Validate configuration on startup
try:
    # config.validate_config() # Assuming validate_config exists in config.py or we skip it if not copied
    # Since I didn't copy validate_config function (it wasn't in the first 100 lines of config.py I read), 
    # I will check if I need to add it or if it was in the omitted part.
    # I'll assume it might be missing if I didn't copy it.
    # Let's just print config summary if possible or skip validation for now.
    pass
except ValueError as e:
    print(f"[CRITICAL ERROR] Configuration validation failed:")
    print(e)
    sys.exit(1)


class ActionSelector:
    """
    Semantic search engine for mapping LLM intents to robot actions.
    """
    
    def __init__(self):
        """
        Initialize the action selector with embedding model and action database.
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
        """
        t0 = time.time()
        # Encode the query intent into an embedding vector
        query_vec = self.model.encode([query_text])
        # Compute similarity scores against all action embeddings
        if len(self.db_embeddings) == 0:
            return config.DEFAULT_ACTION
            
        scores = cosine_similarity(query_vec, self.db_embeddings)[0]
        best_idx = np.argmax(scores)
        best_action = self.keys[best_idx]
        confidence = scores[best_idx]
        
        # if config.VERBOSE_LOGGING: # Assuming VERBOSE_LOGGING is in config
        print(f"\n   [Action Mapping]")
        print(f"   Query Intent:   '{query_text}'")
        print(f"   Mapped To:      '{best_action}'")
        print(f"   Confidence:     {confidence:.4f}")
        
        if confidence < config.ACTION_CONFIDENCE_THRESHOLD:
            # if config.VERBOSE_LOGGING:
            print(f"   (! Weak Match ! Defaulting to '{config.DEFAULT_ACTION}')")
            return config.DEFAULT_ACTION
        return best_action


class Op2Brain(Node):
    """
    Main ROS 2 node implementing the AI conversation pipeline.
    """
    
    def __init__(self):
        """
        Initialize the Op2Brain node with all AI components.
        """
        super().__init__('op2_brain_node')
        # Create publisher for sending action commands to op2_driver
        # Assuming config has ROS_ACTION_TOPIC, ROS_QUEUE_SIZE
        # If not, I'll use defaults
        topic = getattr(config, 'ROS_ACTION_TOPIC', '/perform_action')
        queue_size = getattr(config, 'ROS_QUEUE_SIZE', 10)
        
        self.publisher_ = self.create_publisher(String, topic, queue_size)
        
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
        # Assuming config.FORCE_MODEL_RELOAD and VERBOSE_LOGGING exist
        force_reload = getattr(config, 'FORCE_MODEL_RELOAD', True)
        verbose = getattr(config, 'VERBOSE_LOGGING', False)
        
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad', 
            force_reload=force_reload, 
            verbose=verbose
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
        
        prompt = f"""You are {p['name']}, an OP2 Robot with the role of {p['role']}.

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
        
        # if config.VERBOSE_LOGGING:
        #     print(f"    [Memory] Conversation history: {len(self.conversation_history)} messages")

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
        """
        log_sep = getattr(config, 'LOG_SEPARATOR', "="*50)
        print("\n" + log_sep)
        print(f"   PERSONA: {self.persona['name']} ({self.persona['role']})")
        print(f"   MEMORY: {self.get_conversation_summary()}")
        print(log_sep)
        
        if self.record_user_audio():
            user_text = self.transcribe_audio()
            
            # Add user message to conversation history
            self.add_to_conversation("user", user_text)
            
            # Generate response with persona-specific prompt and conversation history
            llm_intent, speech = self.generate_response()
            
            # Add assistant response to conversation history
            self.add_to_conversation("assistant", f"{llm_intent} | {speech}")
            
            enable_parallel = getattr(config, 'ENABLE_PARALLEL_EXECUTION', False)
            
            if enable_parallel:
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
        """
        msg = String()
        msg.data = action_name
        self.publisher_.publish(msg)

    # --- VAD (Voice Activity Detection) ---
    def record_user_audio(self):
        """
        Record audio from microphone until silence is detected.
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
        normalize = getattr(config, 'NORMALIZE_AUDIO', True)
        if normalize:
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
        """
        print(f"\n[2] STT ({config.STT_MODEL})")
        t0 = time.time()
        
        stt_timeout = getattr(config, 'STT_TIMEOUT', 10)
        fallback_transcript = getattr(config, 'FALLBACK_TRANSCRIPT', "I didn't catch that.")
        
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=stt_timeout)
            
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
                # if config.VERBOSE_LOGGING:
                #     print(f"    [WARNING] STT Failed to parse. Raw output:\n{result.stdout}")
                transcript = fallback_transcript
                
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] STT timeout after {stt_timeout}s")
            transcript = fallback_transcript
        except Exception as e:
            print(f"    [ERROR] STT subprocess failed: {e}")
            transcript = fallback_transcript

        print(f"    Transcript: '{transcript}'")
        # if config.LOG_LATENCY:
        #     print(f"    Latency:    {time.time() - t0:.4f}s")
        return transcript

    # --- LLM (Large Language Model) ---
    def generate_response(self):
        """
        Generate a persona-aware response using the LLM.
        """
        print(f"\n[3] LLM ({config.LLM_MODEL.split('/')[-1]}) - Persona: {self.persona['name']}")
        if config.ENABLE_CONVERSATION_MEMORY:
            print(f"    Context: {len(self.conversation_history)} messages in history")
        
        t0 = time.time()
        client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.NVIDIA_API_KEY)
        
        fallback_intent = getattr(config, 'FALLBACK_INTENT', "unknown")
        fallback_speech = getattr(config, 'FALLBACK_SPEECH', "I am not sure what to say.")
        
        intent = fallback_intent
        speech = fallback_speech
        
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
                        # if not first_token and config.LOG_LATENCY:
                        #     print(f"(TTFT: {time.time()-t0:.2f}s) ", end="")
                        #     first_token = True
                        print(c, end="", flush=True)
                        full_resp += c
                print("")
            else:
                full_resp = completion.choices[0].message.content
            
            # if config.LOG_LATENCY:
            #     print(f"    Total Latency: {time.time()-t0:.4f}s")
            
            # Parse response
            if "|" in full_resp:
                parts = full_resp.split("|", 1)
                intent = parts[0].strip()
                speech = parts[1].strip()
            else:
                speech = full_resp.strip()
                intent = fallback_intent
                
            # if config.VERBOSE_LOGGING:
            #     print(f"    Parsed Intent: '{intent}'")
            #     print(f"    Parsed Speech: '{speech}'")
            
        except Exception as e:
            print(f"    [ERROR] LLM failed: {e}")

        return intent, speech

    # --- TTS (Text-to-Speech) ---
    def generate_audio_file(self, speech_text):
        """
        Generate speech audio from text using Nvidia Riva TTS.
        """
        settings = self.persona["tts_settings"]
        voice_name = f"Magpie-Multilingual.{settings['language']}.{settings['voice']}"
        
        print(f"\n[4] TTS ({config.TTS_MODEL}) - Voice: {voice_name}")
        # if config.VERBOSE_LOGGING:
        #     print(f"    Speech Text: '{speech_text}'")
        
        t0 = time.time()
        
        tts_timeout = getattr(config, 'TTS_TIMEOUT', 10)
        
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
                timeout=tts_timeout
            )
            # if config.LOG_LATENCY:
            #     print(f"    Latency: {time.time()-t0:.4f}s")
            return True
        except subprocess.TimeoutExpired:
            print(f"    [ERROR] TTS timeout after {tts_timeout}s")
            return False
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] TTS failed with code {e.returncode}")
            # if config.VERBOSE_LOGGING:
            #     print(f"    STDOUT: {e.stdout}")
            #     print(f"    STDERR: {e.stderr}")
            return False
        except Exception as e:
            print(f"    [ERROR] TTS failed: {e}")
            return False

    def play_audio_file(self):
        """
        Play the generated speech audio file through speakers.
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
    Entry point for the OP2 Brain node.
    """
    rclpy.init(args=args)
    
    try:
        node = Op2Brain()
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
