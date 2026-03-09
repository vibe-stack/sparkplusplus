import './style.css';
import { getSparkBanner, SPARK_ENGINE_DESCRIPTOR } from '@sparkplusplus/spark';

const app = document.querySelector<HTMLDivElement>('#app');

if (!app) {
  throw new Error('Missing #app root');
}

app.innerHTML = `
  <main class="shell">
    <section class="panel">
      <p class="eyebrow">Bun workspace bootstrap</p>
      <h1>${getSparkBanner()}</h1>
      <p class="body">
        Monorepo scaffold ready. The reusable package lives in <strong>packages/spark</strong>
        and this Vite playground lives in <strong>apps/playground</strong>.
      </p>
      <dl class="facts">
        <div>
          <dt>Package</dt>
          <dd>${SPARK_ENGINE_DESCRIPTOR.name}</dd>
        </div>
        <div>
          <dt>Backend target</dt>
          <dd>${SPARK_ENGINE_DESCRIPTOR.renderer}</dd>
        </div>
        <div>
          <dt>Current stage</dt>
          <dd>${SPARK_ENGINE_DESCRIPTOR.stage}</dd>
        </div>
      </dl>
    </section>
  </main>
`;