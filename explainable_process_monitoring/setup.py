from setuptools import setup, find_packages

setup(name='explainable_process_monitoring',
      packages=find_packages(),
      package_data={
            "predictive_process_monitoring": ["VP/*.py", "prbpm_models/*.py"],
      },
      include_package_data=True,
      version='1.0.5')
