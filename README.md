Matthew's first ML mini project

Original assignment specs by Andy

> Date created: 2025-05-25-Sat
>
> ### Matthew's take-home mini assignment
>
> **Task: build an ML model which, given an image, identifies whether it is a cat 🐈 , hotdog 🌭 , or neither.**
>
> Requirements:
>
> 1. track entire project using `git` and `github` (i should be able to see a commit history full of incremental changes, not just one single repo dump)
> 2. manage dependencies using a tool called `uv` (you've probably never heard of this tool, which is why I'm making you use it. I want to see how you learn a new tool) (`uv` is an amazing tool btw) I should be able to clone your repo, call `uv sync`, and immediately be able to run the project.
> 3. use `PyTorch` to build your model (the entire lab uses `torch`)
> 4. go find your own dataset (or download images from google)
> 5. you are not allowed to have any files ending in `.ipynb` in your repo (yes, I'm banning the use of jupyter notebooks. I have my reasons which I'm happy to explain to you if you're interested)
> 6. keep good documentation. It shouldn't be a pain for me to understand your codebase or you shouldn't need to explain to me how to use your model (that's what the `README` is for)
> 7. if you don't have enough compute, come find me and I can run your training on a lab server
>
> Be creative, I don't care *HOW* you solve this problem. Use any model architecture you'd like, train from scratch or finetune, I don't care.
>
> Try to finish by Wednesday, don't hesitate to ask me questions.
>
> **Most importantly, GLHF!**


## Installation

After cloning this repo, `cd` into it and run:

```sh
uv sync
```

[Install uv](https://docs.astral.sh/uv/)

## Usage

To classify your image or images, run

```sh
uv run main.py
```

and follow the prompts.

You may provide a single image or a directory of images.

*try it with `test`!*

## Acknowledgments

Data comes from Kaggle.


## Original thoughts on how to approach
1. Figure out how to use Github
2. Find dataset for cat/not cat and hotdog/not hotdog
3. follow pytorch tutorial on creating a simple model
4. figure out how to use uv to manage dependencies
5. look up and research more advanced models and fine tune parameters
6. figure out how to deal with more advanced problems (distorted image, when a cat and hotdog is both in an image, etc.)
7. idk
