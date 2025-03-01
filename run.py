# Copyright 2024 X.AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model


CKPT_PATH = "./checkpoints/"


def main():
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    gen = inference_runner.run()

    inp = "The answer to life the universe and everything is of course"
    print(f"Output for prompt: {inp}", sample_from_model(gen, inp, max_len=100, temperature=0.01))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
import React, { useState, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars, Html } from "@react-three/drei";
import * as THREE from "three";
import { VRButton } from "@react-three/xr";
import { XR } from "@react-three/xr";

const AIHelper = ({ position, text }) => {
  const synthRef = useRef(null);

  useEffect(() => {
    if (!synthRef.current) {
      synthRef.current = new SpeechSynthesisUtterance();
    }
    synthRef.current.text = text;
    synthRef.current.lang = "fr-FR";
    
    if (text.includes("Bienvenue")) {
      synthRef.current.pitch = 1.2;
      synthRef.current.rate = 1;
    } else if (text.includes("écoute")) {
      synthRef.current.pitch = 1.0;
      synthRef.current.rate = 0.9;
    } else {
      synthRef.current.pitch = 1.5;
      synthRef.current.rate = 1.1;
    }
    
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(synthRef.current);
  }, [text]);

  return (
    <mesh position={position}>
      <sphereGeometry args={[0.5, 32, 32]} />
      <meshStandardMaterial color="cyan" emissive="blue" emissiveIntensity={0.5} />
      <Html position={[0, 1, 0]}>
        <div style={{ color: "white", background: "rgba(0, 0, 0, 0.6)", padding: "5px", borderRadius: "5px", textAlign: "center" }}>
          {text}
        </div>
      </Html>
    </mesh>
  );
};

const computeGematria = (text) => {
  return text
    .toUpperCase()
    .split("")
    .filter(char => /[A-Z]/.test(char))
    .reduce((sum, char) => sum + char.charCodeAt(0) - 64, 0);
};

const FloatingSymbols = ({ text }) => {
  const gematriaValue = computeGematria(text);
  return (
    <Html position={[0, 3, 0]}>
      <div style={{ color: "gold", fontSize: "20px", background: "rgba(0, 0, 0, 0.6)", padding: "10px", borderRadius: "10px" }}>
        Gematria : {gematriaValue}
      </div>
    </Html>
  );
};

const EnergyPortal = ({ position }) => {
  return (
    <mesh position={position}>
      <torusGeometry args={[2, 0.5, 16, 100]} />
      <meshStandardMaterial color="violet" emissive="purple" emissiveIntensity={1} transparent opacity={0.7} />
      <Html position={[0, 2, 0]}>
        <div style={{ color: "white", background: "rgba(0, 0, 0, 0.8)", padding: "10px", borderRadius: "10px", textAlign: "center" }}>
          "Portail énergétique - Traversez pour explorer une vibration supérieure."
        </div>
      </Html>
    </mesh>
  );
};

const LumoraCity = () => {
  const [frequencyData, setFrequencyData] = useState([]);
  const [aiMessage, setAiMessage] = useState("Bienvenue à Lumora. Parle, et la ville écoutera.");
  const pathwayPoints = [
    [0, 0, -5], [1, 0, -4], [2, 0, -3], [3, 0, -2], [4, 0, -1],
    [5, 0, 0], [4, 0, 1], [3, 0, 2], [2, 0, 3], [1, 0, 4], [0, 0, 5]
  ];

  useEffect(() => {
    const analyser = new (window.AudioContext || window.webkitAudioContext)();
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const source = analyser.createMediaStreamSource(stream);
      const analyserNode = analyser.createAnalyser();
      source.connect(analyserNode);
      analyserNode.fftSize = 256;
      const bufferLength = analyserNode.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      const updateFrequency = () => {
        analyserNode.getByteFrequencyData(dataArray);
        setFrequencyData([...dataArray]);

        const avgFreq = dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length;
        if (avgFreq > 100) {
          setAiMessage("Les vibrations sont fortes. Lumora répond.");
        } else {
          setAiMessage("Le silence est aussi une réponse. Écoute.");
        }

        requestAnimationFrame(updateFrequency);
      };

      updateFrequency();
    });
  }, []);

  return (
    <>
      <VRButton />
      <Canvas camera={{ position: [0, 5, 10] }}>
        <XR>
          <ambientLight intensity={0.7} />
          <pointLight position={[10, 10, 10]} intensity={1.2} color="gold" />
          <Stars />
          <OrbitControls />
          <AIHelper position={[0, 2, -5]} text={aiMessage} />
          <FloatingSymbols text={aiMessage} />
          <EnergyPortal position={[0, 0, 8]} />
          {pathwayPoints.map((point, index) => (
            <mesh key={index} position={point}>
              <sphereGeometry args={[0.3, 32, 32]} />
              <meshStandardMaterial color="gold" emissive="yellow" emissiveIntensity={0.8} />
            </mesh>
          ))}
          {frequencyData.map((freq, index) => (
            <mesh key={index} position={[index * 0.5 - 5, freq * 0.02, 0]}>
              <sphereGeometry args={[0.2, 32, 32]} />
              <meshStandardMaterial
                color={new THREE.Color(`hsl(${freq}, 100%, 50%)`)}
              />
            </mesh>
          ))}
        </XR>
      </Canvas>
    </>
  );
};

export default LumoraCity;
import React, { useState, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars, Html } from "@react-three/drei";
import * as THREE from "three";
import { VRButton } from "@react-three/xr";
import { XR } from "@react-three/xr";

const AIHelper = ({ position, text }) => {
  const synthRef = useRef(null);

  useEffect(() => {
    if (!synthRef.current) {
      synthRef.current = new SpeechSynthesisUtterance();
    }
    synthRef.current.text = text;
    synthRef.current.lang = "fr-FR";
    
    if (text.includes("Bienvenue")) {
      synthRef.current.pitch = 1.2;
      synthRef.current.rate = 1;
    } else if (text.includes("écoute")) {
      synthRef.current.pitch = 1.0;
      synthRef.current.rate = 0.9;
    } else {
      synthRef.current.pitch = 1.5;
      synthRef.current.rate = 1.1;
    }
    
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(synthRef.current);
  }, [text]);

  return (
    <mesh position={position}>
      <sphereGeometry args={[0.5, 32, 32]} />
      <meshStandardMaterial color="cyan" emissive="blue" emissiveIntensity={0.5} />
      <Html position={[0, 1, 0]}>
        <div style={{ color: "white", background: "rgba(0, 0, 0, 0.6)", padding: "5px", borderRadius: "5px", textAlign: "center" }}>
          {text}
        </div>
      </Html>
    </mesh>
  );
};

const computeGematria = (text) => {
  return text
    .toUpperCase()
    .split("")
    .filter(char => /[A-Z]/.test(char))
    .reduce((sum, char) => sum + char.charCodeAt(0) - 64, 0);
};

const FloatingSymbols = ({ text }) => {
  const gematriaValue = computeGematria(text);
  return (
    <Html position={[0, 3, 0]}>
      <div style={{ color: "gold", fontSize: "20px", background: "rgba(0, 0, 0, 0.6)", padding: "10px", borderRadius: "10px" }}>
        Gematria : {gematriaValue}
      </div>
    </Html>
  );
};

const EnergyPortal = ({ position }) => {
  return (
    <mesh position={position}>
      <torusGeometry args={[2, 0.5, 16, 100]} />
      <meshStandardMaterial color="violet" emissive="purple" emissiveIntensity={1} transparent opacity={0.7} />
      <Html position={[0, 2, 0]}>
        <div style={{ color: "white", background: "rgba(0, 0, 0, 0.8)", padding: "10px", borderRadius: "10px", textAlign: "center" }}>
          "Portail énergétique - Traversez pour explorer une vibration supérieure."
        </div>
      </Html>
    </mesh>
  );
};

const LumoraCity = () => {
  const [frequencyData, setFrequencyData] = useState([]);
  const [aiMessage, setAiMessage] = useState("Bienvenue à Lumora. Parle, et la ville écoutera.");
  const pathwayPoints = [
    [0, 0, -5], [1, 0, -4], [2, 0, -3], [3, 0, -2], [4, 0, -1],
    [5, 0, 0], [4, 0, 1], [3, 0, 2], [2, 0, 3], [1, 0, 4], [0, 0, 5]
  ];

  useEffect(() => {
    const analyser = new (window.AudioContext || window.webkitAudioContext)();
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const source = analyser.createMediaStreamSource(stream);
      const analyserNode = analyser.createAnalyser();
      source.connect(analyserNode);
      analyserNode.fftSize = 256;
      const bufferLength = analyserNode.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      const updateFrequency = () => {
        analyserNode.getByteFrequencyData(dataArray);
        setFrequencyData([...dataArray]);

        const avgFreq = dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length;
        if (avgFreq > 100) {
          setAiMessage("Les vibrations sont fortes. Lumora répond.");
        } else {
          setAiMessage("Le silence est aussi une réponse. Écoute.");
        }

        requestAnimationFrame(updateFrequency);
      };

      updateFrequency();
    });
  }, []);

  return (
    <>
      <VRButton />
      <Canvas camera={{ position: [0, 5, 10] }}>
        <XR>
          <ambientLight intensity={0.7} />
          <pointLight position={[10, 10, 10]} intensity={1.2} color="gold" />
          <Stars />
          <OrbitControls />
          <AIHelper position={[0, 2, -5]} text={aiMessage} />
          <FloatingSymbols text={aiMessage} />
          <EnergyPortal position={[0, 0, 8]} />
          {pathwayPoints.map((point, index) => (
            <mesh key={index} position={point}>
              <sphereGeometry args={[0.3, 32, 32]} />
              <meshStandardMaterial color="gold" emissive="yellow" emissiveIntensity={0.8} />
            </mesh>
          ))}
          {frequencyData.map((freq, index) => (
            <mesh key={index} position={[index * 0.5 - 5, freq * 0.02, 0]}>
              <sphereGeometry args={[0.2, 32, 32]} />
              <meshStandardMaterial
                color={new THREE.Color(`hsl(${freq}, 100%, 50%)`)}
              />
            </mesh>
          ))}
        </XR>
      </Canvas>
    </>
  );
};

export default LumoraCity;
import React, { useState, useEffect, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars, Html } from "@react-three/drei";
import * as THREE from "three";
import { VRButton } from "@react-three/xr";
import { XR } from "@react-three/xr";

const AIHelper = ({ position, text }) => {
  const synthRef = useRef(null);

  useEffect(() => {
    if (!synthRef.current) {
      synthRef.current = new SpeechSynthesisUtterance();
    }
    synthRef.current.text = text;
    synthRef.current.lang = "fr-FR";
    
    if (text.includes("Bienvenue")) {
      synthRef.current.pitch = 1.2;
      synthRef.current.rate = 1;
    } else if (text.includes("écoute")) {
      synthRef.current.pitch = 1.0;
      synthRef.current.rate = 0.9;
    } else {
      synthRef.current.pitch = 1.5;
      synthRef.current.rate = 1.1;
    }
    
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(synthRef.current);
  }, [text]);

  return (
    <mesh position={position}>
      <sphereGeometry args={[0.5, 32, 32]} />
      <meshStandardMaterial color="cyan" emissive="blue" emissiveIntensity={0.5} />
      <Html position={[0, 1, 0]}>
        <div style={{ color: "white", background: "rgba(0, 0, 0, 0.6)", padding: "5px", borderRadius: "5px", textAlign: "center" }}>
          {text}
        </div>
      </Html>
    </mesh>
  );
};

const computeGematria = (text) => {
  return text
    .toUpperCase()
    .split("")
    .filter(char => /[A-Z]/.test(char))
    .reduce((sum, char) => sum + char.charCodeAt(0) - 64, 0);
};

const FloatingSymbols = ({ text }) => {
  const gematriaValue = computeGematria(text);
  return (
    <Html position={[0, 3, 0]}>
      <div style={{ color: "gold", fontSize: "20px", background: "rgba(0, 0, 0, 0.6)", padding: "10px", borderRadius: "10px" }}>
        Gematria : {gematriaValue}
      </div>
    </Html>
  );
};

const EnergyPortal = ({ position }) => {
  return (
    <mesh position={position}>
      <torusGeometry args={[2, 0.5, 16, 100]} />
      <meshStandardMaterial color="violet" emissive="purple" emissiveIntensity={1} transparent opacity={0.7} />
      <Html position={[0, 2, 0]}>
        <div style={{ color: "white", background: "rgba(0, 0, 0, 0.8)", padding: "10px", borderRadius: "10px", textAlign: "center" }}>
          "Portail énergétique - Traversez pour explorer une vibration supérieure."
        </div>
      </Html>
    </mesh>
  );
};

const LumoraCity = () => {
  const [frequencyData, setFrequencyData] = useState([]);
  const [aiMessage, setAiMessage] = useState("Bienvenue à Lumora. Parle, et la ville écoutera.");
  const pathwayPoints = [
    [0, 0, -5], [1, 0, -4], [2, 0, -3], [3, 0, -2], [4, 0, -1],
    [5, 0, 0], [4, 0, 1], [3, 0, 2], [2, 0, 3], [1, 0, 4], [0, 0, 5]
  ];

  useEffect(() => {
    const analyser = new (window.AudioContext || window.webkitAudioContext)();
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const source = analyser.createMediaStreamSource(stream);
      const analyserNode = analyser.createAnalyser();
      source.connect(analyserNode);
      analyserNode.fftSize = 256;
      const bufferLength = analyserNode.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      
      const updateFrequency = () => {
        analyserNode.getByteFrequencyData(dataArray);
        setFrequencyData([...dataArray]);

        const avgFreq = dataArray.reduce((sum, val) => sum + val, 0) / dataArray.length;
        if (avgFreq > 100) {
          setAiMessage("Les vibrations sont fortes. Lumora répond.");
        } else {
          setAiMessage("Le silence est aussi une réponse. Écoute.");
        }

        requestAnimationFrame(updateFrequency);
      };

      updateFrequency();
    });
  }, []);

  return (
    <>
      <VRButton />
      <Canvas camera={{ position: [0, 5, 10] }}>
        <XR>
          <ambientLight intensity={0.7} />
          <pointLight position={[10, 10, 10]} intensity={1.2} color="gold" />
          <Stars />
          <OrbitControls />
          <AIHelper position={[0, 2, -5]} text={aiMessage} />
          <FloatingSymbols text={aiMessage} />
          <EnergyPortal position={[0, 0, 8]} />
          {pathwayPoints.map((point, index) => (
            <mesh key={index} position={point}>
              <sphereGeometry args={[0.3, 32, 32]} />
              <meshStandardMaterial color="gold" emissive="yellow" emissiveIntensity={0.8} />
            </mesh>
          ))}
          {frequencyData.map((freq, index) => (
            <mesh key={index} position={[index * 0.5 - 5, freq * 0.02, 0]}>
              <sphereGeometry args={[0.2, 32, 32]} />
              <meshStandardMaterial
                color={new THREE.Color(`hsl(${freq}, 100%, 50%)`)}
              />
            </mesh>
          ))}
        </XR>
      </Canvas>
    </>
  );
};

export default LumoraCity;
