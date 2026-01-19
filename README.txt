========================================
WAV BATCH CONVERTER - AUDIO TRANSCRIPTION TOOL
========================================

This tool automatically converts WAV audio files to searchable text transcriptions.

========================================
ONE-TIME SETUP (15-20 minutes)
========================================

STEP 1: Install Python
-----------------------
1. Go to https://www.python.org/downloads/
2. Download Python 3.10 or newer
3. Run the installer
   IMPORTANT: Check "Add Python to PATH" during installation
4. Verify installation:
   - Windows: Open Command Prompt
   - Mac: Open Terminal
   - Type: python --version
   - You should see: Python 3.x.x

STEP 2: Install Whisper (the transcription software)
----------------------------------------------------
Open Terminal (Mac) or Command Prompt (Windows) and run:

    pip install openai-whisper

This will take 5-10 minutes to download and install.

STEP 3: Unzip This Folder
--------------------------
1. Unzip "wav_batch_converter.zip" to your Desktop or Documents
2. You should see:
   - wavConverter.py (the main script)
   - README.txt (this file)
   - wav_files/ (empty folder)
   - transcriptions/ (empty folder)

========================================
HOW TO USE (Each Time You Need Transcriptions)
========================================

STEP 1: Add Your Audio Files
-----------------------------
Copy all your WAV files into the "wav_files" folder

STEP 2: Run the Script
----------------------
1. Open Terminal (Mac) or Command Prompt (Windows)
2. Navigate to this folder:
   
   cd Desktop/wav_batch_converter
   
   (Adjust the path if you put it somewhere else)

3. Run the script:
   
   python wavConverter.py

STEP 3: Wait for Processing
----------------------------
- You'll see progress updates in the terminal
- The script shows: [X/Total] Processing: filename.wav
- You can close your laptop lid - it will keep running
- For 1000 files of ~5 minutes each: expect 2-6 hours total

STEP 4: Get Your Results
-------------------------
Open the "transcriptions" folder. You'll find three subfolders:

1. plain_text/
   - filename_transcript.txt - Simple text transcription
   
2. timestamped/
   - filename_timestamped.txt - Text with timestamps for each segment
   
3. json/
   - filename_data.json - Complete data including timestamps and metadata

========================================
RESUMING AFTER INTERRUPTION
========================================

If the script stops or crashes:
- Just run "python wavConverter.py" again
- It automatically skips files that are already done
- It picks up where it left off

========================================
TROUBLESHOOTING
========================================

Problem: "No WAV files found"
Solution: Make sure your files are in the "wav_files" folder

Problem: Warning message "FP16 is not supported on CPU"
Solution: This is normal and can be safely ignored. The script automatically 
uses FP32 processing instead, which works perfectly fine on regular computers.
You may see this warning appear multiple times - it's just informational.

Problem: Script crashes or errors
Solution: 
1. Check the log file in transcriptions/transcription_log_[date].txt
2. Contact me with the error message

Problem: Need to retry failed files
Solution:
1. Open wavConverter.py in a text editor
2. Find this line (near the bottom):
   # transcriber.retry_failed()
3. Remove the # at the start
4. Save and run again

Problem: Want better accuracy (but slower processing)
Solution:
1. Open wavConverter.py in a text editor
2. Find this line:
   MODEL_SIZE = "base"
3. Change to:
   MODEL_SIZE = "small"   (2x slower, more accurate)
   or
   MODEL_SIZE = "medium"  (4x slower, very accurate)
4. Save and run again

========================================
OUTPUT EXPLANATION
========================================

For each WAV file, you get 3 output files:

1. filename_transcript.txt
   - Plain text, easy to search
   - Use this for keyword searching

2. filename_timestamped.txt
   - Same text with timestamps
   - Format: [00:02:15 -> 00:02:23] Text here
   - Use this to find exact moments in the recording

3. filename_data.json
   - Complete technical data
   - Includes word-level details
   - Use this for advanced analysis

========================================
COST
========================================

This solution is FREE - it runs on your computer.
No per-minute charges, no API costs.

========================================
CONTACT
========================================

If you have issues or questions, contact me with:
- Screenshot of any error messages
- The log file from transcriptions/transcription_log_[date].txt

Good luck with your transcriptions!
