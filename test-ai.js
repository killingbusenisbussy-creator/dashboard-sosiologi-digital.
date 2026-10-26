require('dotenv').config();
const { GoogleGenAI } = require('@google/genai');

const ai = new GoogleGenAI({});

async function generate(modelName, prompt, systemInstruction) {
  return await ai.models.generateContent({
    model: modelName,
    contents: prompt,
    config: {
      systemInstruction: systemInstruction,
    },
  });
}

async function main() {
  const systemInstruction = 'Senior Tech Startup Mentor yang kritis dan to the point';
  const defaultPrompt = 'Apa ide bisnis mikro-SaaS berbasis AI yang paling cepat menghasilkan uang tahun ini?';

  // Membaca teks pertanyaan dari argumen terminal (process.argv[2]) atau fallback ke default
  const prompt = (process.argv[2] ? process.argv.slice(2).join(' ') : null) || defaultPrompt;

  console.log(`Pertanyaan: "${prompt}"\n`);

  let model = 'gemini-3.8-flash';

  try {
    const response = await generate(model, prompt, systemInstruction);
    console.log(`[Model: ${model}]\n`);
    console.log(response.text);
  } catch (error) {
    // Jika gemini-3.8-flash mengalami kendala kuota/high demand (503) atau 404, beralih ke Flash aktif
    console.warn(`[Catatan] Model '${model}' tidak dapat diakses (${error.status || 'error'}). Beralih ke versi Flash terbaru yang aktif...`);
    const fallbackModel = 'gemini-3.5-flash-lite';
    try {
      const fallbackResponse = await generate(fallbackModel, prompt, systemInstruction);
      console.log(`[Model: ${fallbackModel}]\n`);
      console.log(fallbackResponse.text);
    } catch (fallbackError) {
      console.error('Terjadi kesalahan:', fallbackError.message);
    }
  }
}

main();
