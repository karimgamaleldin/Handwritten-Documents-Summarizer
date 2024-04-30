from transformers import pipeline


class Summarizer:
    def __init__(self, model="facebook/bart-large-cnn"):
        self.model = model
        self.summarizer = pipeline("summarization", model=model, framework="pt")

    def summarize(self, text, max_length=100, min_length=30):
        if self.model == "facebook/bart-large-cnn":
            return self.summarizer(
                text, max_length=max_length, min_length=min_length, do_sample=False
            )[0]["summary_text"]
        else:
            return self.summarizer(text, max_length=max_length, min_length=min_length)[0][
                "summary_text"
            ]


# summ = Summarizer()
# ARTICLE = """ 
#  In the 21st century, the global energy sector has seen a significant shift towards renewable energy sources. This transition is driven by the urgent need to reduce greenhouse gas emissions and combat climate change. Renewable energy technologies, such as solar panels, wind turbines, and hydroelectric power, offer sustainable alternatives to traditional fossil fuels. Solar energy has become increasingly popular due to the declining cost of photovoltaic cells and the simplicity of installation. Households and businesses worldwide are adopting solar panels to generate clean energy and reduce electricity costs. Similarly wind energy has grown in adoption with massive wind farms being constructed both offshore and onshore. These wind farms can produce vast amounts of electricity, contributing significantly to national grids Hydroelectric power remains a dominant renewable energy source, particularly in countries with extensive river systems. It harnesses the power of flowing water to generate electricity, providing a reliable and constant energy supply. Unlike solar and wind energy hydroelectric power is not subject to fluctuations in weather conditions, making it a cornerstone for many renewable energy strategies Despite the benefits, the transition to renewable energy is not without challenges The variability of solar and wind energy requires enhancements in grid technology and energy storage solutions to ensure a steady supply of power. Furthermore, there are environmental and social impacts associated with the construction of large-scale renewable energy installations, such as habitat disruption and displacement of communities. In conclusion, while renewable energy technologies are pivotal in the fight against climate change, their deployment must be managed carefully to mitigate adverse effects on communities and the environment. The future of global energy will likely depend on a balanced mix of renewable sources, coupled with innovative technologies to improve energy efficiency and reduce overall consumption
# """

# summary = summ.summarize(ARTICLE, min_length=100, max_length=150)

# print(summary)
# print(len(summary))
# print(len(summary.split(" ")))
