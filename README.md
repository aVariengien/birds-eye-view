# 🦉 Bird's Eye View


<img src="img/plan-of-Imola.jpg" alt="drawing" width="500"/>

_A city plan of Imola by Leonardo Da Vinci_.

## Introduction

In August 1502, Leonardo Da Vinci, was appointed by Cesare Borgia to survey the city of Imola to strengthen its fortifications. 

He walked along the streets of the city, took measurements, used some clever geometry, and his drawing genius to produce an accurate view of the city, as seen from above.

This single image compresses countless on-the-ground observations. Once done, the same plan can be used for many activities, from urban planning to simply finding your way in the city. Anyone can use it, even people who have no clue how to produce the plan in the first place.

The field of cartography gives us a collection of representations of the physical world in which we live at different scales, from your neighborhood to the whole planet. With satellite imagery and aerial photography, we can directly _see_ shapes previously constructed from the ground.

**What map do we have of our _information_ world?**

Every day, we are bombarded with mountains of text, images, and videos. Our information world might have taken a larger space in our lives. To digest this flow, we, well, read the text, sometimes partially, look at the images, and watch the videos. We do the equivalent of mindlessly walking in the streets of Imola, looking at every wall, sometimes running to get to a building, hoping that after long enough we'll learn to navigate the city. 

But how can we hope to understand the fast-evolving information neighborhood in which we live? The world?

We have cool statisics like [Google Trends](https://trends.google.com/trends/), or [Google Ngram Viewer](https://books.google.com/ngrams/), hand-made meta-analysis or crappy summaries from ChatGPT.

Fondamentally, we don't have good representations of the information world that would free us from engaging with the content at the smallest scale.

The goal of **🦉 Bird's Eye View** is to create a maps of this space, enabling you to form an understanding of a large corpus of text at a high level.

## Setup

```
pip install -r requirements.txt
touch cache/cache.json
streamlit run viz.py
```

You need to have `OPENAI_API_KEY` defined in your environment variables. I advise adding `export OPENAI_API_KEY=sk-...` in your `.bashrc`.

The project is in very early stage, and will probably show errors, e.g. if you tick `Use qualitative colors` but display a continuous value (e.g. search results).

## Walkthrough

The maps are created by dividing each text into chunks, and using OpenAI embeddings models followed by UMAP to reduce the dimension. You can customize the pipeline in the configuration side pannel.

The tool supports importing text from webpages, or local and online PDF files. Webpages are preferred as they contain style information.

Here is an example of a map of the last 
[Zvi Mowshowwitz's newsletters](https://thezvi.substack.com/).

![image](img/map-zvi.png)

Each point is a chunk. We can click on the chunk to see its content. The dark orange line points to the next chunk in the text, the light orange line to the previous. _View source_ links to the chunk in the original webpage.

## Example Workflow

When loading a new corpus, I'd advise setting the `n_neighbors` parameter of UMAP low (15-20). The visualization will focus on local structure, creating bundles of points.

Your goal is to understand: what are these clusters about? If they are clearly together, there is something that they have in common.

To understand what, focus on one cluster. Read the chunks, and make a guess. 

Let's focus on this cluster.

![image](img/focus-cluster.png)

From reading a few chunks, it seems to talk about jailbreaks. Let's make a search.

![image](img/first-search.png)

The search embedds the prompt, and computes the dot product with all the embeddings in the corpus. It applies a sharpen filter on the results to filter the noisy values around 0.1, and boost the significantly high values. The `threshold` controls the sharpen function.

OK, the guess was good! It seems to light up the points in this cluster, but also some points in the middle of the map. Let's look at one of these outliers.

![image](img/negative-point.png)

It also mentions jailbreaking. From reading the points in the cluster, the text seems to emphasize how easy it is to jailbreak models, whereas this point is mentioning jailbreaking in another context. 

Let's refine the prompt, we'll try `All AI models can be jailbroken`.

![image](img/second-search.png)

It seems a bit more specific, the previous outlier point lights up much less.

You got it, the goal is to engage in a back and forth to understand what precisely makes this cluster unique. 

After understanding the content of many clusters, one can increase `n_neighbors` and look at the global structure of the map.



## Future Use Cases

* After understanding a space, upload a new text you don't know. Understand this new text in comparison to the space you know. Look at a given text through many different prism.
* Analyzing thousands of generations of LLM. This could enable a loom with a high branching factor but low depth. Such a tool can be useful in brainstorming.
* Scaling up, making a map of a continent of information.
* Expert chess player engages in _chuncking_, ie. recongizing at a glace a certain configuration of the board. Could the back-and-forth described above enable text-chunking specialized for a corpus? Can I digest text faster if I have a mental model of the possible chunks I can find?