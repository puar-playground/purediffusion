from setuptools import setup

setup(
    name='purediffusion',
    version='0.1.0',
    description='A torch implementation for DDPM with DDIM sampling.',
    url='https://github.com/puar-playground/purediffusion',
    author='Jian Chen',
    author_email='cjvault1223@gmail.com',
    license='MIT',
    packages=['purediffusion'],
    install_requires=['torch',
                      'numpy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)

