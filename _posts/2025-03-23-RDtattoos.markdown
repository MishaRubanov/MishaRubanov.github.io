---
layout: post
title:  "Reaction-Diffusion Tattoos"
date:   2025-03-23 12:00:45 -0600
categories: jekyll update
---
For the last few months, I've wanted to write a basic Turing pattern simulator as a tattoo - representing the beauty of these reaction diffusion patterns in life. I would like this to be a dot pattern as well. This barebones repository can be found at this [repository](https://github.com/MishaRubanov/RDtattoo). A lot of this effort was inspired and adapted from this [repo](https://github.com/ijmbarr/turing-patterns/blob/master/turing-patterns.ipynb).


Alternatively, using the filename directly:

You can view the [Tattoo Notebook]({{ site.baseurl }}/jupyter/tattoo.html).

<iframe src="/_jupyter/tattoo.html" width="100%" height="800px" title="Tattoo Notebook"></iframe>

<iframe
  src="https://jupyterlite.github.io/demo/repl/index.html?kernel=python&toolbar=1"
  width="100%"
  height="500px"
>
</iframe>

<!-- Add the slider -->
<div>
  <label for="slider">Rate this post:</label>
  <input type="range" id="slider" name="slider" min="1" max="10" value="5" oninput="updateSliderValue(this.value)">
  <span id="sliderValue">5</span>
</div>

<!-- Add JavaScript to handle the slider value -->
<script>
function updateSliderValue(value) {
  document.getElementById('sliderValue').innerText = value;
}
</script>

<iframe
  src="https://rdtattoos.streamlit.app/?embed=true"
  style="height: 450px; width: 100%;"
></iframe>