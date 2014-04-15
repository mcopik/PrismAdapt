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

/**
 * Main interface for kernel components.
 * Specifies methods to access includes, type/method etc. declarations and created source code.
 *
 */
public interface KernelComponent
{
	/**
	 * 
	 * @return true if component has additional include
	 */
	boolean hasIncludes();

	/**
	 * 
	 * @return true if component has global declaration
	 */
	boolean hasDeclaration();

	/**
	 * 
	 * @return list of additional includes
	 */
	List<Include> getIncludes();

	/**
	 * @return declaration source code (type definition, method definition)
	 */
	KernelComponent getDeclaration();

	/**
	 * Accepts visitor to change internal state.
	 * @param visitor object
	 */
	void accept(VisitorInterface v);

	/**
	 * 
	 * @return component source code
	 */
	String getSource();

}