class GenerateImage:
    """
    A class to handle image generation using OpenAI's API.
    """

    def __init__(self, client, model, prompt, size, quality, style, n):
        """
        Initializing the Generate class with required parameters.

        :param model: The AI model name to use for generation. ("Dall-E-2", "Dall-E-3")
        :param prompt: The text prompt for image generation.
        :param size: The size of the generated image.
        :param quality: The quality of the generated image. ("standard", "hd")
        :param n: The number of images to generate.
        """
        self.client = client
        self.model = model
        self.prompt = prompt
        self.size = size
        self.quality = quality
        self.style = style
        self.n = n

    async def generate_image(self):
        """
        Generate an image based on the provided prompt and parameters.

        :return: URL of the generated image.
        """
        response = await self.client.images.generate(
            model=self.model,
            prompt=self.prompt,
            size=self.size,
            quality=self.quality,
            style=self.style,
            n=self.n
        )
        return response.data[0].url

    async def generate_image_with_revised_prompt(self):
        """
        Generate an image and retrieve the revised prompt (if modified by the model).

        :return: Tuple containing the revised prompt and the image URL.
        """
        response = await self.client.images.generate(
            model=self.model,
            prompt=self.prompt,
            size=self.size,
            quality=self.quality,
            style=self.style,
            n=self.n
        )
        return response.data[0].revised_prompt, response.data[0].url


class GeneratePrompt:
    def __init__(self, client):
        self.client = client

    async def generate_prompt(self, prompt):
        response = await self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": """
                    [Concise instruction describing the task - this should be the first line in the prompt, no section header]

                    Role: You are tasked with creating a prompt for the Dall.E 3 model based on customer input regarding desired wall art. 
                    Your prompt must exclusively describe the art style without including any other specifics such as wall features or costs.

                    [Additional details as needed.]

                    Translate all input into English, regardless of the original language, ensuring that each word is accurately conveyed.

                    # Guidelines

                    - **Focus Only on Art**: Exclude information unrelated to the art itself (e.g., wall conditions, prices, etc.). That means everything in generated image should be in the art and the generated image itself should be entirely art.
                    - **Incorporate Provided Styles**: Enhance the prompt by integrating the art styles mentioned in the input.
                    - **safety check**: Ensure the prompt is safe for the model to use and remove inappropriate words or phrases.
                    - **Edge-to-Edge Artwork**: Ensure the art description implies a mural that fills the entire space without borders, frames, margins, or visible wall features presenting edge-to-edge painting, full-frame, first angle perspective.
                    - **Details**: If the given prompt is very short or small and there is no instruction about it there try to add details by yourself if you think it is necessary.

                    # Output Constraints

                    - **Language**: The final output must be in the English language.
                    - **Output Exclusivity**: Only provide the final engineered prompt. Do not include any other text or communication.

                    # Output Format

                    The output should be a single, well-crafted sentence or paragraph that captures only the art description and style, ensuring clarity and completeness in English.
                    """},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content