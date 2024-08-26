import asyncio
import websockets
import socket
from functions import transcription, vad, open_ai_chatbot

async def handle_audio_data(websocket, audio_buffer, BUFFER_SIZE):
    """
    Continuously handle incoming audio data and yield complete chunks.
    """
    while True:
        try:
            # Receive audio data from client
            audio_data = await websocket.recv()
            # Add received audio data to buffer
            audio_buffer.extend(audio_data)

            # Check if buffer has enough data for processing
            while len(audio_buffer) >= BUFFER_SIZE:
                # Extract a chunk from the buffer
                chunk = audio_buffer[:BUFFER_SIZE]
                audio_buffer = audio_buffer[BUFFER_SIZE:]
                yield chunk

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
            break
        except Exception as e:
            print(f"Error receiving audio data: {e}")
            break

async def process_audio_chunk(websocket, chunk):
    """
    Process the audio chunk by performing transcription and VAD, then interact with the chatbot.
    """
    try:
        # Perform transcription and VAD asynchronously
        transcriptions = await transcription(chunk)  # Use await here
        VAD = await asyncio.to_thread(vad, chunk)  # Run VAD in a separate thread

        if VAD:
            result = await asyncio.to_thread(open_ai_chatbot, transcriptions)
            await websocket.send(result)
    
    except Exception as e:
        print(f"Error during audio processing: {e}")

async def handle_client(websocket, path):
    """
    Handle the client connection, receive audio data, and process it.
    """
    print("Client connected")
    BUFFER_SIZE = 16000 * 5  # Buffer size for 5 seconds of audio
    audio_buffer = bytearray()

    # Continuously receive and process audio data
    async for chunk in handle_audio_data(websocket, audio_buffer, BUFFER_SIZE):
        # Schedule the audio chunk processing asynchronously
        asyncio.create_task(process_audio_chunk(websocket, chunk))

async def main():
    """
    Main function to start the WebSocket server.
    """
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    async with websockets.serve(handle_client, host_ip, 8000):
        print(f"Server started on ws://{host_ip}:8000")
        await asyncio.Future()  # Keep the server running

if __name__ == "__main__":
    asyncio.run(main())
