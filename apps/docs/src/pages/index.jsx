import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Hero from '@site/src/components/Hero';
import Features from '@site/src/components/Features';
import Stats from '@site/src/components/Stats';

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} - ${siteConfig.tagline}`}
      description="Transform any educational problem into beautiful animated explanations. Support for Math, Chemistry, Physics, and Computer Science.">
      <Hero />
      <main>
        <Features />
        <Stats />
      </main>
    </Layout>
  );
}
