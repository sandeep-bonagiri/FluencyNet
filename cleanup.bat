@echo off
echo 🧹 Cleaning up unnecessary files...

del agent_client.py 2>nul
del index.html 2>nul
del therapy_knowledge_base.yml 2>nul
del vocal_agent_mac.sh 2>nul
del evaluate_stt.py 2>nul
del finetune_whisper.py 2>nul
del setup_test_data.py 2>nul
del requirements.txt 2>nul

echo ✅ Cleanup complete.
echo ℹ️  requirements.txt was removed to prevent conflicting dependency installation.
echo    Dependencies are now managed directly in download_requirements.py.
pause