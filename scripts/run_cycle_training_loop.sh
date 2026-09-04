function STMResetAndRun()
{
    rm -rf "chroma_db_stm" && python hqe_trainer.py
}

for i in {1..81}; do STMResetAndRun || break; done