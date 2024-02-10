# Train GPT
Training the model for Ben Miriello 2024 portfolio site.

## How it's done

You'll need to generate your own `dataset.csv` file in the root directory with prompt and response rows. You'll want a lot of rows to get decent results, but you can start out with a small set for a test run.

```
prompt,response
Are we not men?,We are devo.
Should I stay or should I go?,"If you go, there will be trouble. If you stay it will be double."
```

Install packages
``` bash
pip install torch transformers pandas accelerate
```

Then train the model
``` bash
python train_model.py
```

Once that succeeds you can test it 
``` bash
python test_model.py -p "Write your own test question here"
```

## Using Docker

in project root directory, build and run the docker image:
``` bash
docker build -t gpt-training-image .

docker run -v gpt-training-image`
```

Or to run and save outside the docker container:
``` bash
docker run -v ./trained_model:/app/trained_model gpt-training-image
```

## Setting up on runpod

If you don't have a gpu this may be the solution for you.

Start a new pod without a template. You don't need much GPU power to run these operations so pick the cheapest option.

### Optional: generate ssh keys to access your pod.

Runpod gives instructions under 'Configure Public Key' in the connect menu for your pod on the Pods page.

Generate a new key:
``` bash
ssh-keygen -t ed25519 -C "your_email@example.com"`

cat ~/.ssh/id_ed25519.pub
```

Add the public key that's returned to your runpod SSH Public Keys under Settings.

Run the command runpod gives under your pod's Connect menu 'Connection Options', something like

``` bash
ssh <your pod id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```
If that doesn't get you into a shell with your pod, refer to the runpod documentation and instructions.


