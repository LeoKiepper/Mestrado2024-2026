import os
import sys
import tempfile

# Detect --noplot in argv (simple, robust)
NOPLOT = any(arg == "--noplot" for arg in sys.argv)

def report_output(value):
    """
    Se existe a variável de ambiente MEASURE_OUTPUT_PATH, escreve o número
    neste ficheiro (apenas um token, one-line), atomically via os.replace.
    Caso contrário, escreve no stdout (comportamento de fallback).
    """
    # formata de forma compacta e robusta
    try:
        s = f"{float(value):.12g}"
    except Exception:
        s = "nan"

    out_path = os.environ.get("MEASURE_OUTPUT_PATH")
    if out_path:
        # escreve atomically: escreve para tmp dentro do mesmo dir e substitui
        d = os.path.dirname(out_path) or "."
        fd, tmpname = tempfile.mkstemp(prefix=".tmp_measure_", dir=d)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tf:
                tf.write(s + "\n")
                tf.flush()
                os.fsync(tf.fileno())
            # substitute atomically
            os.replace(tmpname, out_path)
        except Exception:
            # fallback para stdout se algo falhar
            try:
                if os.path.exists(tmpname):
                    os.remove(tmpname)
            except Exception:
                pass
            sys.stdout.write(s + "\n")
            sys.stdout.flush()
    else:
        # comportamento antigo: print no stdout
        sys.stdout.write(s + "\n")
        sys.stdout.flush()