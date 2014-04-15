//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package simulator.gpu.opencl.kernel.expression;

import java.util.List;

public class Include implements KernelComponent
{
	public final String includePath;
	public final boolean isLocal;

	public Include(String path, boolean isLocal)
	{
		includePath = path;
		this.isLocal = isLocal;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.KernelComponent#hasInclude()
	 */
	@Override
	public boolean hasIncludes()
	{
		return false;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.KernelComponent#hasDeclaration()
	 */
	@Override
	public boolean hasDeclaration()
	{
		return false;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.KernelComponent#getInclude()
	 */
	@Override
	public List<Include> getIncludes()
	{
		return null;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.KernelComponent#getDeclaration()
	 */
	@Override
	public KernelComponent getDeclaration()
	{
		return null;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.KernelComponent#getSource()
	 */
	@Override
	public String getSource()
	{
		StringBuilder builder = new StringBuilder("#include ");
		if (isLocal) {
			builder.append("\"");
		} else {
			builder.append("<");
		}
		builder.append(includePath);
		if (isLocal) {
			builder.append("\"");
		} else {
			builder.append(">");
		}
		return builder.toString();
	}

	@Override
	public void accept(VisitorInterface v)
	{
		//do nothing
	}
}