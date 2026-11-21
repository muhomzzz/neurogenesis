# Neurogenesis (first prototype)
Run: `pip install numpy pytest`, `python -m pytest -q test_engine.py` (7 tests), `python server.py`, open http://localhost:8000.

## Architecture
- `engine.py`: 320 Izhikevich spiking neurons (excitatory regular-spiking, inhibitory fast-spiking), current-based synapses,
  six regions (SENS, CTX, HIPP, AMY, BG, PFC) wired by explicit pathways. CTX->BG synapses learn by reward-modulated STDP
  (STDP eligibility trace x dopamine = reward prediction error) with soft homeostatic scaling.
- Neuromodulators: DA (RPE), NE (arousal: noise/exploration), ACh (uncertainty: sensory gain), 5-HT (tracked, not yet functional).
- Nociception-like path: harm injects current into AMY; threat raises arousal and memory salience.
- Memory: episodic store with sparse hippocampal index and emotion-weighted encoding, forgetting decay, retrieval with
  reconsolidation, sleep-like replay building a semantic table, PFC recurrence (working memory, untested), save/load.
- `server.py` + `web/index.html`: controls, spike raster, region rates, modulators, memories, learning curve, neuron inspector.

## Measured evidence
Over 6 seeds, correct-choice rate rose from about 0.54 (first 40 trials) to 0.75 (last 40); chance is 0.5. Single seeds are noisy.

## Approximations and open problems
- Not a human brain emulation; regions are functional analogies at ~320 neurons; the approach/avoid task is a toy.
- Credit assignment is masked to the chosen BG group (engineering shortcut); the value table is a symbolic critic.
- Episodic/semantic stores are symbolic; only CTX->BG learning is synaptic. No language, spatial, cerebellar, or forgetting-protection tests yet.
- Pain variables are computational signals, not evidence of experience.
- The web page was served and its API checked, but not viewed in a browser.

## Roadmap
Neural critic and cortex-hippocampus plasticity; cerebellar prediction and a learned world model; retention/interference tests;
vectorized/GPU scaling; per-region ablations; experiment log and architecture search.
