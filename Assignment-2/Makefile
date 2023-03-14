PYTHON          := python3.9

SRC             := src
PYCACHE         := $(SRC)/__pycache__

DATA_DIR        := training_samples
SAMPLES         := $(shell find $(DATA_DIR) -type f)
MODEL           := sound_classifer.sklearn

LIVE            := $(SRC)/live.py
MODEL_GEN       := $(SRC)/model_gen.py
SOUND_CLASS     := $(SRC)/sound_sample.py

all: gen_model

$(MODEL): $(SAMPLES) $(MODEL_GEN) $(SOUND_CLASS)
	$(PYTHON) $(MODEL_GEN) -o $(MODEL) -s $(DATA_DIR)

# Alias for model.
gen_model: $(MODEL)

.PHONY: run clean

run: $(MODEL) $(LIVE)
	$(PYTHON) $(LIVE) -m $(MODEL)

clean:
	rm -rf $(PYCACHE) $(MODEL)
